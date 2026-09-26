#include "Utilities/Common/FilesystemDurability.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

namespace Thor::FilesystemDurability {
namespace {

class ScopedFd {
   public:
    explicit ScopedFd(int fd) : fd(fd) {}
    ScopedFd(const ScopedFd&) = delete;
    ScopedFd& operator=(const ScopedFd&) = delete;
    ~ScopedFd() {
        if (fd >= 0) {
            ::close(fd);
        }
    }

    [[nodiscard]] int get() const { return fd; }

   private:
    int fd;
};

[[nodiscard]] std::filesystem::path normalizedDirectory(const std::filesystem::path& directory) {
    if (directory.empty()) {
        return std::filesystem::path(".");
    }
    return directory;
}

[[nodiscard]] std::filesystem::path parentDirectory(const std::filesystem::path& path) {
    return normalizedDirectory(path.parent_path());
}

[[noreturn]] void throwErrno(const char* operation,
                             const std::filesystem::path& path,
                             int errorNumber) {
    throw std::runtime_error(std::string(operation) + " failed for '" + path.string() + "': " +
                             std::strerror(errorNumber));
}

void syncOpenedDescriptor(int fd,
                          const char* operation,
                          const std::filesystem::path& path) {
    if (::fsync(fd) != 0) {
        const int errorNumber = errno;
        throwErrno(operation, path, errorNumber);
    }
}

[[nodiscard]] bool sameDirectoryPath(const std::filesystem::path& lhs,
                                     const std::filesystem::path& rhs) {
    return normalizedDirectory(lhs).lexically_normal() ==
           normalizedDirectory(rhs).lexically_normal();
}

}  // namespace

void syncFile(const std::filesystem::path& path) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        const int errorNumber = errno;
        throwErrno("open for fsync", path, errorNumber);
    }
    ScopedFd scopedFd(fd);
    syncOpenedDescriptor(scopedFd.get(), "fsync", path);
}

void syncDirectory(const std::filesystem::path& directory) {
    const std::filesystem::path resolvedDirectory = normalizedDirectory(directory);
    const int fd = ::open(resolvedDirectory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd < 0) {
        const int errorNumber = errno;
        throwErrno("open directory for fsync", resolvedDirectory, errorNumber);
    }
    ScopedFd scopedFd(fd);
    syncOpenedDescriptor(scopedFd.get(), "directory fsync", resolvedDirectory);
}

void syncParentDirectory(const std::filesystem::path& path) {
    syncDirectory(parentDirectory(path));
}

void createDirectoriesDurably(const std::filesystem::path& directory) {
    const std::filesystem::path target = normalizedDirectory(directory);

    std::error_code errorCode;
    if (std::filesystem::exists(target, errorCode)) {
        if (errorCode) {
            throw std::runtime_error("Failed to inspect directory '" + target.string() + "': " +
                                     errorCode.message());
        }
        if (!std::filesystem::is_directory(target, errorCode) || errorCode) {
            throw std::runtime_error("Filesystem path exists but is not a directory: '" +
                                     target.string() + "'.");
        }
        return;
    }
    if (errorCode) {
        throw std::runtime_error("Failed to inspect directory '" + target.string() + "': " +
                                 errorCode.message());
    }

    std::vector<std::filesystem::path> missing;
    std::filesystem::path cursor = target;
    while (true) {
        errorCode.clear();
        if (std::filesystem::exists(cursor, errorCode)) {
            if (errorCode) {
                throw std::runtime_error("Failed to inspect directory ancestor '" + cursor.string() +
                                         "': " + errorCode.message());
            }
            break;
        }
        if (errorCode) {
            throw std::runtime_error("Failed to inspect directory ancestor '" + cursor.string() +
                                     "': " + errorCode.message());
        }

        missing.push_back(cursor);
        const std::filesystem::path parent = parentDirectory(cursor);
        if (sameDirectoryPath(parent, cursor)) {
            throw std::runtime_error("Unable to find an existing ancestor while creating directory '" +
                                     target.string() + "'.");
        }
        cursor = parent;
    }

    errorCode.clear();
    if (!std::filesystem::is_directory(cursor, errorCode) || errorCode) {
        throw std::runtime_error("Directory ancestor is not a directory: '" + cursor.string() + "'.");
    }

    for (auto it = missing.rbegin(); it != missing.rend(); ++it) {
        errorCode.clear();
        const bool created = std::filesystem::create_directory(*it, errorCode);
        if (errorCode) {
            throw std::runtime_error("Failed to create directory '" + it->string() + "': " +
                                     errorCode.message());
        }
        if (!created) {
            errorCode.clear();
            if (!std::filesystem::is_directory(*it, errorCode) || errorCode) {
                throw std::runtime_error("Filesystem path appeared while creating directories but is not a directory: '" +
                                         it->string() + "'.");
            }
        }
        syncParentDirectory(*it);
    }
}

void durableRename(const std::filesystem::path& source,
                   const std::filesystem::path& destination) {
    std::error_code errorCode;
    std::filesystem::rename(source, destination, errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to rename '" + source.string() + "' to '" +
                                 destination.string() + "': " + errorCode.message());
    }

    const std::filesystem::path sourceParent = parentDirectory(source);
    const std::filesystem::path destinationParent = parentDirectory(destination);

    // For a cross-directory rename, persist the new link before explicitly
    // persisting removal of the old one. This mirrors checkpoint
    // create-before-destroy ordering at the filesystem namespace layer.
    syncDirectory(destinationParent);
    if (!sameDirectoryPath(sourceParent, destinationParent)) {
        syncDirectory(sourceParent);
    }
}

void durableRemoveAll(const std::filesystem::path& path) {
    std::error_code errorCode;
    const bool exists = std::filesystem::exists(path, errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to inspect path before removal '" + path.string() + "': " +
                                 errorCode.message());
    }
    if (!exists) {
        return;
    }

    std::filesystem::remove_all(path, errorCode);
    if (errorCode) {
        throw std::runtime_error("Failed to remove path '" + path.string() + "': " + errorCode.message());
    }
    syncParentDirectory(path);
}

}  // namespace Thor::FilesystemDurability
