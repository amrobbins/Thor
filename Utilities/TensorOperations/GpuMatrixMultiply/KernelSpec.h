#pragma once

#include "Utilities/Common/Stream.h"
#include "Utilities/TensorOperations/GpuMatrixMultiply/CublasKernelRequirement.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <unordered_map>
#include <utility>

using std::string;

/**
 * Every kernel must specify at the minimum:
 *  1. aRowsPerBlock
 *  2. bColsPerBlock
 *  3. executeKernel
 *
 * Additionally, if the kernel has any of the other available requirements/characteristics, then the kernel must also specify those.
 */
struct KernelWithSpec {
    int aRowsPerBlock;
    int bColsPerBlock;
    int blocksKSplitInto;

    // A value of <= 1 indicates no requirement
    int aRowSizeModulusRequirement;
    int aColSizeModulusRequirement;
    int bRowSizeModulusRequirement;
    int bColSizeModulusRequirement;

    KernelWithSpec() {
        // The following fields must be assigned
        aRowsPerBlock = -1;
        bColsPerBlock = -1;
        executeKernel = ERROR_launchKernelFunctionWasNotAssigned;

        // The following fields may be left at default when applicable
        aRowSizeModulusRequirement = 0;
        aColSizeModulusRequirement = 0;
        bRowSizeModulusRequirement = 0;
        bColSizeModulusRequirement = 0;
        blocksKSplitInto = 1;
        getWorkspaceSize = hasNoWorkspace;
    }

    void (*executeKernel)(const half *A,
                          const half *B,
                          half *C,
                          half *workspace,
                          int32_t A_rows,
                          int32_t A_cols,
                          int32_t B_cols,
                          int32_t ld_A,
                          int32_t ld_B,
                          int32_t ld_C,
                          Stream stream);

    unsigned int (*getWorkspaceSize)(KernelRequirement kernelRequirement);

    // Once a kernel index is given to a kernel, it refers to that kernel forever.
    // Once a kernel index gets into a release, it can never be changed.
    // If the kernel gets deprecated, the index is not reused.
    // naming convention: _ARowsPerBlock_BColsPerBlock_Description
    enum class KernelIndex {
        _256_96_bigSharedBlockA16Restrict = 100,
        _256_96_bigSharedBlock = 101,
        _128_96_bigSharedBlockA16Restrict = 102,
        _128_96_bigSharedBlock = 103,
        _64_96_bigSharedBlockA16Restrict = 104,
        _64_96_bigSharedBlock = 105,
        _32_96_bigSharedBlockA16Restrict = 106,
        _32_96_bigSharedBlock = 107,
        _16_96_bigSharedBlockA16Restrict = 108,
        _16_96_bigSharedBlock = 109,
        _256_80_bigSharedBlockA16Restrict = 110,
        _256_80_bigSharedBlock = 111,
        _128_80_bigSharedBlockA16Restrict = 112,
        _128_80_bigSharedBlock = 113,
        _64_80_bigSharedBlockA16Restrict = 114,
        _64_80_bigSharedBlock = 115,
        _32_80_bigSharedBlockA16Restrict = 116,
        _32_80_bigSharedBlock = 117,
        _16_80_bigSharedBlockA16Restrict = 118,
        _16_80_bigSharedBlock = 119,
        _256_64_bigSharedBlockA16Restrict = 120,
        _256_64_bigSharedBlock = 121,
        _128_64_bigSharedBlockA16Restrict = 122,
        _128_64_bigSharedBlock = 123,
        _64_64_bigSharedBlockA16Restrict = 124,
        _64_64_bigSharedBlock = 125,
        _32_64_bigSharedBlockA16Restrict = 126,
        _32_64_bigSharedBlock = 127,
        _16_64_bigSharedBlockA16Restrict = 128,
        _16_64_bigSharedBlock = 129,
        _80_32_AB16Restrict = 130,
        _80_32_A16Restrict = 131,
        _80_32_loadToShared = 132,
        _64_32_AB16Restrict = 133,
        _64_32_A16Restrict = 134,
        _64_32_loadToShared = 135,
        _48_32_AB16Restrict = 136,
        _48_32_A16Restrict = 137,
        _48_32_loadToShared = 138,
        _32_32_AB16Restrict = 139,
        _32_32_A16Restrict = 140,
        _32_32_loadToShared = 141,
        _16_32_AB16Restrict = 142,
        _16_32_A16Restrict = 143,
        _16_32_loadToShared = 144,
        _256_32_bigSharedBlockA16Restrict = 145,
        _256_32_bigSharedBlock = 146,
        _128_32_bigSharedBlockA16Restrict = 147,
        _128_32_bigSharedBlock = 148,
        _64_32_bigSharedBlockA16Restrict = 149,
        _64_32_bigSharedBlock = 150,
        _32_32_bigSharedBlockA16Restrict = 151,
        _32_32_bigSharedBlock = 152,
        _16_32_bigSharedBlockA16Restrict = 153,
        _16_32_bigSharedBlock = 154,
        _16_16_ABRestrict = 155,
        _16_16_ARestrict = 156,
        _32_16_ARestrict = 157,
        _64_16_ARestrict = 158,
        _128_16_ARestrict = 159,
        _16_16_loadToShared = 160,
        _32_16_loadToShared = 161,
        _64_16_loadToShared = 162,
        _128_16_loadToShared = 163,
        _32_8_ABRestrict = 164,
        _32_8_ARestrict = 165,
        _64_8_ARestrict = 166,
        _128_8_ARestrict = 167,
        _256_8_ARestrict = 168,
        _32_8_loadToShared = 169,
        _64_8_loadToShared = 170,
        _128_8_loadToShared = 171,
        _256_8_loadToShared = 172,
        _128_112_slimSharedBlockA16Restrict = 173,
        _128_112_slimSharedBlock = 174,
        _256_96_bigSharedBlockA16Restrict_reduce2 = 175,
        _256_96_bigSharedBlockA16Restrict_reduce4 = 176,
        _256_96_bigSharedBlockA16Restrict_reduce8 = 177,
        _256_96_bigSharedBlockA16Restrict_reduce6 = 178,
        _256_96_bigSharedBlock_reduce2 = 179,
        _256_96_bigSharedBlock_reduce4 = 180,
        _256_96_bigSharedBlock_reduce8 = 181,
        _256_96_bigSharedBlock_reduce6 = 182,
        _256_80_bigSharedBlockA16Restrict_reduce2 = 183,
        _256_80_bigSharedBlockA16Restrict_reduce4 = 184,
        _256_80_bigSharedBlockA16Restrict_reduce8 = 185,
        _256_80_bigSharedBlockA16Restrict_reduce6 = 186,
        _256_80_bigSharedBlock_reduce2 = 187,
        _256_80_bigSharedBlock_reduce4 = 188,
        _256_80_bigSharedBlock_reduce8 = 189,
        _256_80_bigSharedBlock_reduce6 = 190,
        _256_64_bigSharedBlockA16Restrict_reduce2 = 191,
        _256_64_bigSharedBlockA16Restrict_reduce4 = 192,
        _256_64_bigSharedBlockA16Restrict_reduce8 = 193,
        _256_64_bigSharedBlockA16Restrict_reduce6 = 194,
        _256_64_bigSharedBlock_reduce2 = 195,
        _256_64_bigSharedBlock_reduce4 = 196,
        _256_64_bigSharedBlock_reduce8 = 197,
        _256_64_bigSharedBlock_reduce6 = 198,
        _128_96_bigSharedBlockA16Restrict_reduce2 = 199,
        _128_96_bigSharedBlockA16Restrict_reduce4 = 200,
        _128_96_bigSharedBlockA16Restrict_reduce8 = 201,
        _128_96_bigSharedBlockA16Restrict_reduce6 = 202,
        _128_96_bigSharedBlock_reduce2 = 203,
        _128_96_bigSharedBlock_reduce4 = 204,
        _128_96_bigSharedBlock_reduce8 = 205,
        _128_96_bigSharedBlock_reduce6 = 206,
        _128_80_bigSharedBlockA16Restrict_reduce2 = 207,
        _128_80_bigSharedBlockA16Restrict_reduce4 = 208,
        _128_80_bigSharedBlockA16Restrict_reduce8 = 209,
        _128_80_bigSharedBlockA16Restrict_reduce6 = 210,
        _128_80_bigSharedBlock_reduce2 = 211,
        _128_80_bigSharedBlock_reduce4 = 212,
        _128_80_bigSharedBlock_reduce8 = 213,
        _128_80_bigSharedBlock_reduce6 = 214,
        _128_64_bigSharedBlockA16Restrict_reduce2 = 215,
        _128_64_bigSharedBlockA16Restrict_reduce4 = 216,
        _128_64_bigSharedBlockA16Restrict_reduce8 = 217,
        _128_64_bigSharedBlockA16Restrict_reduce6 = 218,
        _128_64_bigSharedBlock_reduce2 = 219,
        _128_64_bigSharedBlock_reduce4 = 220,
        _128_64_bigSharedBlock_reduce8 = 221,
        _128_64_bigSharedBlock_reduce6 = 222,
        _256_48_bigSharedBlockA16Restrict_reduce2 = 227,
        _256_48_bigSharedBlockA16Restrict_reduce4 = 228,
        _256_48_bigSharedBlockA16Restrict_reduce8 = 229,
        _256_48_bigSharedBlockA16Restrict_reduce6 = 230,
        _256_48_bigSharedBlock_reduce2 = 231,
        _256_48_bigSharedBlock_reduce4 = 232,
        _256_48_bigSharedBlock_reduce8 = 233,
        _256_48_bigSharedBlock_reduce6 = 234,
        _256_32_bigSharedBlockA16Restrict_reduce2 = 235,
        _256_32_bigSharedBlockA16Restrict_reduce4 = 236,
        _256_32_bigSharedBlockA16Restrict_reduce8 = 237,
        _256_32_bigSharedBlockA16Restrict_reduce6 = 238,
        _256_32_bigSharedBlock_reduce2 = 239,
        _256_32_bigSharedBlock_reduce4 = 240,
        _256_32_bigSharedBlock_reduce8 = 241,
        _256_32_bigSharedBlock_reduce6 = 242,
        _128_48_bigSharedBlockA16Restrict_reduce2 = 243,
        _128_48_bigSharedBlockA16Restrict_reduce4 = 244,
        _128_48_bigSharedBlockA16Restrict_reduce8 = 245,
        _128_48_bigSharedBlockA16Restrict_reduce6 = 246,
        _128_48_bigSharedBlock_reduce2 = 247,
        _128_48_bigSharedBlock_reduce4 = 248,
        _128_48_bigSharedBlock_reduce8 = 249,
        _128_48_bigSharedBlock_reduce6 = 250,
        _128_32_bigSharedBlockA16Restrict_reduce2 = 251,
        _128_32_bigSharedBlockA16Restrict_reduce4 = 252,
        _128_32_bigSharedBlockA16Restrict_reduce8 = 253,
        _128_32_bigSharedBlockA16Restrict_reduce6 = 254,
        _128_32_bigSharedBlock_reduce2 = 255,
        _128_32_bigSharedBlock_reduce4 = 256,
        _128_32_bigSharedBlock_reduce8 = 257,
        _128_32_bigSharedBlock_reduce6 = 258,
        _16_48_AB16Restrict = 259,
        _32_48_AB16Restrict = 260,
        _48_48_AB16Restrict = 261,
        _48_48_A16Restrict = 262,
        _48_48_noRestrict = 263,
        _64_48_AB16Restrict = 264,
        _128_48_A16Restrict = 265,
        _128_48_noRestrict = 266,
        _256_48_A16Restrict = 267,
        _256_48_noRestrict = 268,
        _64_48_A16Restrict = 269,
        _64_48_noRestrict = 270,
        _32_48_A16Restrict = 271,
        _32_48_noRestrict = 272,
        _16_48_A16Restrict = 273,
        _16_48_noRestrict = 274,
    };

    KernelIndex id;

   private:
    static unsigned int hasNoWorkspace(KernelRequirement kernelRequirement) { return 0; }

    static void ERROR_launchKernelFunctionWasNotAssigned(const half *A,
                                                         const half *B,
                                                         half *C,
                                                         half *workspace,
                                                         int32_t A_rows,
                                                         int32_t A_cols,
                                                         int32_t B_cols,
                                                         int32_t ld_A,
                                                         int32_t ld_B,
                                                         int32_t ld_C,
                                                         Stream stream) {
        assert(false);
    }
};
