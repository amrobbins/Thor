#pragma once

#include <filesystem>

namespace Thor::FilesystemDurability {

// Force the current contents/metadata of an existing regular file to stable
// storage. Throws std::runtime_error on failure.
void syncFile(const std::filesystem::path& path);

// Force directory-entry updates in an existing directory to stable storage.
// Linux requires this in addition to fsync'ing the file itself when a caller
// needs create/rename/remove operations to survive an abrupt machine failure.
void syncDirectory(const std::filesystem::path& directory);

void syncParentDirectory(const std::filesystem::path& path);

// Create every missing path component and make each newly-created directory
// entry durable in its parent before proceeding to the next component.
void createDirectoriesDurably(const std::filesystem::path& directory);

// Atomically rename source to destination and then make the rename durable.
// When the rename crosses directories, the destination parent is synced first
// so a crash cannot make the source removal durable before the new name is.
void durableRename(const std::filesystem::path& source,
                   const std::filesystem::path& destination);

// Remove a file/directory tree and make removal of its top-level directory
// entry durable in the parent.
void durableRemoveAll(const std::filesystem::path& path);

}  // namespace Thor::FilesystemDurability
