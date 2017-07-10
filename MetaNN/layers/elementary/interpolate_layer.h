#pragma once
#include <MetaNN/facilities/named_params.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>

namespace MetaNN
{
using InterpolateLayerInput = NamedParams<struct InterpolateLayerWeight1,
                                          struct InterpolateLayerWeight2,
                                          struct InterpolateLayerLambda>;

namespace NSInterpolateLayer
{
template <bool isFeedbackout>
struct FeedbackOut_
{
    template <typename ElementType, typename DeviceType>
    using InternalType = LayerTraits::LayerInternalBufType<ElementType, DeviceType,
                                                           CategoryTags::Matrix>;

    template <typename T1, typename T2, typename TL, typename DataType>
    static void DataRecord(const T1& p_1, const T2& p_2, const TL& p_l,
                           DataType& p1, DataType& p2, DataType& plambda)
    {
        p1.push(MakeDynamic(p_1));
        p2.push(MakeDynamic(p_2));
        plambda.push(MakeDynamic(p_l));
    }

    template <typename TGrad, typename DataType>
    static auto FeedBack(const TGrad& grad, DataType& p1,
                         DataType& p2, DataType& plambda)
    {
        if ((p1.empty()) || (p2.empty()) || (plambda.empty()))
        {
            throw std::runtime_error("Cannot do FeedBackward for InterpolateLayer");
        }
        auto tmp = grad.template Get<LayerIO>();
        auto res1 = plambda.top() * tmp;
        auto res2 = (1 - plambda.top()) * tmp;
        auto res_lambda = (p1.top() - p2.top()) * tmp;
        auto res = InterpolateLayerInput::Create().template Set<InterpolateLayerWeight1>(std::move(res1))
                                              .template Set<InterpolateLayerWeight2>(std::move(res2))
                                              .template Set<InterpolateLayerLambda>(std::move(res_lambda));
        plambda.pop();
        p1.pop();
        p2.pop();
        return res;
    }
};

template <>
struct FeedbackOut_<false>
{
    template <typename ElementType, typename DeviceType>
    using InternalType = NullParameter;

    template <typename T1, typename T2, typename TL, typename DataType>
    static void DataRecord(const T1&, const T2&, const TL&,
                           DataType&, DataType&, DataType&){}

    template <typename TGrad, typename DataType>
    static auto FeedBack(const TGrad&, const DataType&,
                         const DataType&, const DataType&)
    {
        return InterpolateLayerInput::Create();
    }
};
}

template <typename TPolicies>
class InterpolateLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;

public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
    static constexpr bool IsUpdate = false;
    using InputType = InterpolateLayerInput;
    using OutputType = LayerIO;

private:
    using ElementType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::ElementType;
    using DeviceType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::DeviceType;

private:
    using FeedbackOut_ = NSInterpolateLayer::FeedbackOut_<IsFeedbackOutput>;
    using DataType = typename FeedbackOut_::template InternalType<ElementType,
                                                                  DeviceType>;
public:
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val1 = p_in.template Get<InterpolateLayerWeight1>();
        const auto& val2 = p_in.template Get<InterpolateLayerWeight2>();
        const auto& val_lambda = p_in.template Get<InterpolateLayerLambda>();

        using rawType1 = std::decay_t<decltype(val1)>;
        using rawType2 = std::decay_t<decltype(val2)>;
        using rawType3 = std::decay_t<decltype(val_lambda)>;

        static_assert(!std::is_same<rawType1, NullParameter>::value, "parameter1 is invalid");
        static_assert(!std::is_same<rawType2, NullParameter>::value, "parameter2 is invalid");
        static_assert(!std::is_same<rawType3, NullParameter>::value, "parameter lambda is invalid");

        FeedbackOut_::DataRecord(val1, val2, val_lambda,
                                 m_weight1, m_weight2, m_weight_lambda);
        return LayerIO::Create().template Set<LayerIO>(Interpolate(val1, val2, val_lambda));
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        return FeedbackOut_::FeedBack(p_grad, m_weight1, m_weight2, m_weight_lambda);
    }

    void NeutralInvariant()
    {
        LayerTraits::NeutralInvariant(m_weight1);
        LayerTraits::NeutralInvariant(m_weight2);
        LayerTraits::NeutralInvariant(m_weight_lambda);
    }

private:
    DataType m_weight1;
    DataType m_weight2;
    DataType m_weight_lambda;
};
}
