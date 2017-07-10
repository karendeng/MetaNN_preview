#include "test_named_params.h"
#include <iostream>
#include <cassert>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
struct A; struct B; struct Weight; struct XX;

struct FParams : public NamedParams<struct A,
                                    struct B,
                                    struct Weight> {};

template <typename TIn>
float fun(const TIn& in) {
    auto a = in.template Get<A>();
    auto b = in.template Get<B>();
    auto weight = in.template Get<Weight>();

    return a * weight + b * (1 - weight);
}
}

void test_named_params()
{
    cout << "Test named params...\t";
    auto res = fun(FParams::Create()
                             .Set<A>(1.3f)
                             .Set<B>(2.4f)
                             .Set<Weight>(0.1f));
    assert(fabs(res - 0.1 * 1.3 - 0.9 * 2.4) < 0.0001);
    cout << "done\n";
}
