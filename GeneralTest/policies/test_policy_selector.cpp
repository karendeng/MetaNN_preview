#include "test_policy_selector.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <set>
#include <MetaNN/meta_nn.h>
using namespace std;
using namespace MetaNN;

namespace
{
struct AccPolicy
{
    struct AccuEnum
    {
        struct Add;
        struct Mul;
    };
    using Accu = AccuEnum::Add;

    struct IsAveValue;
    static constexpr bool IsAve = false;

    struct ValueType;
    using Value = float;
};

#include <MetaNN/policies/policy_macro_begin.h>
EnumPolicyObj (PAddAccu,     AccPolicy, Accu,  Add);
EnumPolicyObj (PMulAccu,     AccPolicy, Accu,  Mul);
ValuePolicyObj(PAve,         AccPolicy, IsAve, true);
ValuePolicyObj(PNoAve,       AccPolicy, IsAve, false);
TypePolicyObj (PFloatValue,  AccPolicy, Value, float);
TypePolicyObj (PDoubleValue, AccPolicy, Value, double);
#include <MetaNN/policies/policy_macro_end.h>

template <typename...TPolicies>
struct Accumulator
{
    using TPoliCont = PolicyContainer<TPolicies...>;
    using TPolicyRes = PolicySelect<AccPolicy, TPoliCont>;

    using ValueType = typename TPolicyRes::Value;
    static constexpr bool is_ave = TPolicyRes::IsAve;
    using AccuType = typename TPolicyRes::Accu;

    using AAA = AccPolicy::AccuEnum::Add;
    using AAM = AccPolicy::AccuEnum::Mul;

    template <typename TAccu, typename TIn,
              std::enable_if_t<std::is_same<TAccu, AAA>::value>* = nullptr>
    static auto imp(const TIn& in)
    {
        ValueType count = 0;
        ValueType res = 0;
        for (const auto& x : in)
        {
            res += x;
            count += 1;
        }

        if (is_ave)
        {
            res /= count;
        }
        return res;
    }

    template <typename TAccu, typename TIn,
              std::enable_if_t<std::is_same<TAccu, AAM>::value>* = nullptr>
    static auto imp(const TIn& in)
    {
        ValueType res = 1;
        ValueType count = 0;
        for (const auto& x : in)
        {
            res *= x;
            count += 1;
        }
        if (is_ave)
        {
            res = pow(res, 1.0 / count);
        }
        return res;
    }

public:
    template <typename TIn>
    static auto Eval(const TIn& in)
    {
        return imp<AccuType>(in);
    }
};
}

void test_policy_selector()
{
    cout << "Test policy selector...\t";

    const int a[] = {1, 2, 3, 4, 5};
    assert(fabs(Accumulator<>::Eval(a) - 15) < 0.0001);
    assert(fabs(Accumulator<PMulAccu>::Eval(a) - 120) < 0.0001);
    assert(fabs(Accumulator<PMulAccu, PAve>::Eval(a) - pow(120.0, 0.2)) < 0.0001);
    assert(fabs(Accumulator<PAve, PMulAccu>::Eval(a) - pow(120.0, 0.2)) < 0.0001);

    cout << "done" << endl;
}