#include "Utilities/Common/FilesystemDurability.h"

#include "gtest/gtest.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path uniqueTempPath(const std::string& prefix) {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           (prefix + "_" + std::to_string(nonce));
}

class TempTree {
   public:
    explicit TempTree(const std::string& prefix) : root(uniqueTempPath(prefix)) {
        std::filesystem::remove_all(root);
    }
    ~TempTree() {
        std::error_code ignored;
        std::filesystem::remove_all(root, ignored);
    }

    std::filesystem::path root;
};

}  // namespace

TEST(FilesystemDurability, CreatesNestedDirectoriesAndSyncsFiles) {
    TempTree tree("thor_filesystem_durability_create");
    const std::filesystem::path nested = tree.root / "a" / "b" / "c";

    ASSERT_NO_THROW(Thor::FilesystemDurability::createDirectoriesDurably(nested));
    ASSERT_TRUE(std::filesystem::is_directory(nested));

    const std::filesystem::path file = nested / "payload.bin";
    {
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.good());
        out << "durable payload";
    }

    EXPECT_NO_THROW(Thor::FilesystemDurability::syncFile(file));
    EXPECT_NO_THROW(Thor::FilesystemDurability::syncDirectory(nested));
    EXPECT_NO_THROW(Thor::FilesystemDurability::syncParentDirectory(file));
}

TEST(FilesystemDurability, DurableRenameWorksWithinOneDirectory) {
    TempTree tree("thor_filesystem_durability_same_dir_rename");
    Thor::FilesystemDurability::createDirectoriesDurably(tree.root);

    const std::filesystem::path source = tree.root / "candidate.tmp";
    const std::filesystem::path destination = tree.root / "candidate";
    {
        std::ofstream out(source, std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.good());
        out << "checkpoint";
    }
    Thor::FilesystemDurability::syncFile(source);

    ASSERT_NO_THROW(Thor::FilesystemDurability::durableRename(source, destination));
    EXPECT_FALSE(std::filesystem::exists(source));
    EXPECT_TRUE(std::filesystem::is_regular_file(destination));
}

TEST(FilesystemDurability, DurableRenameWorksAcrossDirectories) {
    TempTree tree("thor_filesystem_durability_cross_dir_rename");
    const std::filesystem::path sourceParent = tree.root / "source";
    const std::filesystem::path destinationParent = tree.root / "destination";
    Thor::FilesystemDurability::createDirectoriesDurably(sourceParent);
    Thor::FilesystemDurability::createDirectoriesDurably(destinationParent);

    const std::filesystem::path source = sourceParent / "best";
    Thor::FilesystemDurability::createDirectoriesDurably(source);
    {
        std::ofstream out(source / "model.thor.tar", std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.good());
        out << "checkpoint";
    }
    Thor::FilesystemDurability::syncFile(source / "model.thor.tar");
    Thor::FilesystemDurability::syncDirectory(source);

    const std::filesystem::path destination = destinationParent / "best";
    ASSERT_NO_THROW(Thor::FilesystemDurability::durableRename(source, destination));
    EXPECT_FALSE(std::filesystem::exists(source));
    EXPECT_TRUE(std::filesystem::is_regular_file(destination / "model.thor.tar"));
}

TEST(FilesystemDurability, DurableRemoveAllRemovesPublishedTree) {
    TempTree tree("thor_filesystem_durability_remove");
    const std::filesystem::path victim = tree.root / "old_checkpoint";
    Thor::FilesystemDurability::createDirectoriesDurably(victim);
    {
        std::ofstream out(victim / "payload", std::ios::binary | std::ios::trunc);
        ASSERT_TRUE(out.good());
        out << "old";
    }

    ASSERT_NO_THROW(Thor::FilesystemDurability::durableRemoveAll(victim));
    EXPECT_FALSE(std::filesystem::exists(victim));
}
