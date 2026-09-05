| # | Stage | Operator | Output shape | Kernel | Stride | Dil. | Params | MACs | Act. MiB |
|---:|---|---|---|---|---:|---:|---:|---:|---:|
| 0 | feature extraction (left branch) | Conv | 1x32x184x616 | 5x5 | 2x2 | 1x1 | 2,432 | 272,025,600 | 13.84 |
| 1 | feature extraction (left branch) | Conv | 1x32x92x308 | 5x5 | 2x2 | 1x1 | 25,632 | 725,401,600 | 3.46 |
| 2 | feature extraction (left branch) | Conv | 1x32x46x154 | 5x5 | 2x2 | 1x1 | 25,632 | 181,350,400 | 0.86 |
| 3 | feature extraction (left branch) | Conv | 1x32x23x77 | 5x5 | 2x2 | 1x1 | 25,632 | 45,337,600 | 0.22 |
| 4 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 5 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 6 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 7 | feature extraction (left branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 8 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 9 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 10 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 11 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 12 | feature extraction (left branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 13 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 14 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 15 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 16 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 17 | feature extraction (left branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 18 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 19 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 20 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 21 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 22 | feature extraction (left branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 23 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 24 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 25 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 26 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 27 | feature extraction (left branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 28 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 29 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 30 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 31 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 32 | feature extraction (left branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 33 | feature extraction (left branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 34 | feature extraction (left branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | 9,248 | 16,321,536 | 0.22 |
| 35 | feature extraction (right branch) | Conv | 1x32x184x616 | 5x5 | 2x2 | 1x1 | shared | 272,025,600 | 13.84 |
| 36 | feature extraction (right branch) | Conv | 1x32x92x308 | 5x5 | 2x2 | 1x1 | shared | 725,401,600 | 3.46 |
| 37 | feature extraction (right branch) | Conv | 1x32x46x154 | 5x5 | 2x2 | 1x1 | shared | 181,350,400 | 0.86 |
| 38 | feature extraction (right branch) | Conv | 1x32x23x77 | 5x5 | 2x2 | 1x1 | shared | 45,337,600 | 0.22 |
| 39 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 40 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 41 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 42 | feature extraction (right branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 43 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 44 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 45 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 46 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 47 | feature extraction (right branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 48 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 49 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 50 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 51 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 52 | feature extraction (right branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 53 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 54 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 55 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 56 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 57 | feature extraction (right branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 58 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 59 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 60 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 61 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 62 | feature extraction (right branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 63 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 64 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 65 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 66 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 67 | feature extraction (right branch) | Add | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 68 | feature extraction (right branch) | LeakyRelu | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 69 | feature extraction (right branch) | Conv | 1x32x23x77 | 3x3 | 1x1 | 1x1 | shared | 16,321,536 | 0.22 |
| 70 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 71 | cost volume construction | Concat | 1x32x23x78 | - | - | - | - | 0 | 0.22 |
| 72 | cost volume construction | Slice | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 73 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 74 | cost volume construction | Concat | 1x32x23x79 | - | - | - | - | 0 | 0.22 |
| 75 | cost volume construction | Slice | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 76 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 77 | cost volume construction | Concat | 1x32x23x80 | - | - | - | - | 0 | 0.22 |
| 78 | cost volume construction | Slice | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 79 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 80 | cost volume construction | Concat | 1x32x23x81 | - | - | - | - | 0 | 0.23 |
| 81 | cost volume construction | Slice | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 82 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 83 | cost volume construction | Concat | 1x32x23x82 | - | - | - | - | 0 | 0.23 |
| 84 | cost volume construction | Slice | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 85 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 86 | cost volume construction | Concat | 1x32x23x83 | - | - | - | - | 0 | 0.23 |
| 87 | cost volume construction | Slice | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 88 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 89 | cost volume construction | Concat | 1x32x23x84 | - | - | - | - | 0 | 0.24 |
| 90 | cost volume construction | Slice | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 91 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 92 | cost volume construction | Concat | 1x32x23x85 | - | - | - | - | 0 | 0.24 |
| 93 | cost volume construction | Slice | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 94 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 95 | cost volume construction | Concat | 1x32x23x86 | - | - | - | - | 0 | 0.24 |
| 96 | cost volume construction | Slice | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 97 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 98 | cost volume construction | Concat | 1x32x23x87 | - | - | - | - | 0 | 0.24 |
| 99 | cost volume construction | Slice | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 100 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 101 | cost volume construction | Concat | 1x32x23x88 | - | - | - | - | 0 | 0.25 |
| 102 | cost volume construction | Slice | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 103 | cost volume construction | Sub | 1x32x23x77 | - | - | - | - | 0 | 0.22 |
| 104-115 | cost volume construction | Unsqueeze x12 | 1x1x32x23x77 | - | - | - | 0 | 2.59 |
| 116 | cost volume construction | Concat | 1x12x32x23x77 | - | - | - | - | 0 | 2.59 |
| 117 | cost volume construction | Transpose | 1x32x12x23x77 | - | - | - | - | 0 | 2.59 |
| 118 | cost volume aggregation (3D) | Conv | 1x32x12x23x77 | 3x3x3 | 1x1x1 | 1x1x1 | 27,680 | 587,575,296 | 2.59 |
| 119 | cost volume aggregation (3D) | LeakyRelu | 1x32x12x23x77 | - | - | - | - | 0 | 2.59 |
| 120 | cost volume aggregation (3D) | Conv | 1x32x12x23x77 | 3x3x3 | 1x1x1 | 1x1x1 | 27,680 | 587,575,296 | 2.59 |
| 121 | cost volume aggregation (3D) | LeakyRelu | 1x32x12x23x77 | - | - | - | - | 0 | 2.59 |
| 122 | cost volume aggregation (3D) | Conv | 1x32x12x23x77 | 3x3x3 | 1x1x1 | 1x1x1 | 27,680 | 587,575,296 | 2.59 |
| 123 | cost volume aggregation (3D) | LeakyRelu | 1x32x12x23x77 | - | - | - | - | 0 | 2.59 |
| 124 | cost volume aggregation (3D) | Conv | 1x32x12x23x77 | 3x3x3 | 1x1x1 | 1x1x1 | 27,680 | 587,575,296 | 2.59 |
| 125 | cost volume aggregation (3D) | LeakyRelu | 1x32x12x23x77 | - | - | - | - | 0 | 2.59 |
| 126 | cost volume aggregation (3D) | Conv | 1x1x12x23x77 | 3x3x3 | 1x1x1 | 1x1x1 | 865 | 18,361,728 | 0.08 |
| 127 | cost volume aggregation (3D) | Squeeze | 1x12x23x77 | - | - | - | - | 0 | 0.08 |
| 128 | upsample + soft-argmin | Resize | 1x12x368x1232 | - | - | - | - | 0 | 20.75 |
| 129 | upsample + soft-argmin | Neg | 1x12x368x1232 | - | - | - | - | 0 | 20.75 |
| 130 | upsample + soft-argmin | Softmax | 1x12x368x1232 | - | - | - | - | 0 | 20.75 |
| 131 | upsample + soft-argmin | Mul | 1x12x368x1232 | - | - | - | - | 5,440,512 | 20.75 |
| 132 | upsample + soft-argmin | ReduceSum | 1x1x368x1232 | - | - | - | - | 0 | 1.73 |
| 133 | refinement (full resolution) | Concat | 1x4x368x1232 | - | - | - | - | 0 | 6.92 |
| 134 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 1x1 | 1,184 | 522,289,152 | 55.34 |
| 135 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 1x1 | 9,248 | 4,178,313,216 | 55.34 |
| 136 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 137 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 1x1 | 9,248 | 4,178,313,216 | 55.34 |
| 138 | refinement (full resolution) | Add | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 139 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 140 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 2x2 | 9,248 | 4,178,313,216 | 55.34 |
| 141 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 142 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 2x2 | 9,248 | 4,178,313,216 | 55.34 |
| 143 | refinement (full resolution) | Add | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 144 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 145 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 4x4 | 9,248 | 4,178,313,216 | 55.34 |
| 146 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 147 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 4x4 | 9,248 | 4,178,313,216 | 55.34 |
| 148 | refinement (full resolution) | Add | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 149 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 150 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 8x8 | 9,248 | 4,178,313,216 | 55.34 |
| 151 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 152 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 8x8 | 9,248 | 4,178,313,216 | 55.34 |
| 153 | refinement (full resolution) | Add | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 154 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 155 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 1x1 | 9,248 | 4,178,313,216 | 55.34 |
| 156 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 157 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 1x1 | 9,248 | 4,178,313,216 | 55.34 |
| 158 | refinement (full resolution) | Add | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 159 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 160 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 1x1 | 9,248 | 4,178,313,216 | 55.34 |
| 161 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 162 | refinement (full resolution) | Conv | 1x32x368x1232 | 3x3 | 1x1 | 1x1 | 9,248 | 4,178,313,216 | 55.34 |
| 163 | refinement (full resolution) | Add | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 164 | refinement (full resolution) | LeakyRelu | 1x32x368x1232 | - | - | - | - | 0 | 55.34 |
| 165 | refinement (full resolution) | Conv | 1x1x368x1232 | 3x3 | 1x1 | 1x1 | 289 | 130,572,288 | 1.73 |
| 166 | residual add + ReLU | Add | 1x1x368x1232 | - | - | - | - | 0 | 1.73 |
| 167 | residual add + ReLU | Relu | 1x1x368x1232 | - | - | - | - | 0 | 1.73 |
