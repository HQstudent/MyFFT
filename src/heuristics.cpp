#include "heuristics.h"

namespace myfft {

PathKind SelectPath1d(int nX, int /*nBatch*/) {
    if (nX > 0 && (nX & (nX - 1)) == 0) {
        return PathKind::Pow2SingleKernel;
    }
    /* M3:混合基(radix 2/3/5/7 组合)与 Bluestein 在此分派 */
    return PathKind::Unsupported;
}

} // namespace myfft
