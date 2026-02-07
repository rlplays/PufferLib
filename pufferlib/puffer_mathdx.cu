#include <cublasdx.hpp>
using namespace cublasdx;

using GEMM =  decltype(Size<32 /* m */, 32 /* n */, 32 /* k */>()
                      + Precision<float>()
                      + Type<type::real>()
                      + Function<function::MM>()
                      + Arrangement<cublasdx::row_major /* A */, cublasdx::col_major /* B */>());
