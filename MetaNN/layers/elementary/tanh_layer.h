#pragma once
#include <MetaNN/facilities/named_params.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>

namespace MetaNN
{
namespace NSTanhLayer
{
template <bool isFeedback>
struct FeedbackOut_
{
    template <typename ElementType, typename DeviceType>
    using InternalType = LayerTraits::LayerInternalBufType<ElementType, DeviceType,
                                                           CategoryTags::Matrix>;
    template <typename T, typename DataType>
    static void RecordData(const T& p_in, DataType& data)
    {
        data.push(MakeDynamic(p_in));
    }

    template <typename TGrad, typename DataType>
    static auto FeedBack(DataType& data, const TGrad& grad)
    {
        if (data.empty())
        {
            throw std::runtime_error("Cannot feed back in SigmoidLayer");
        }
        auto tmp = grad.template Get<LayerIO>();
        auto& tmp2 = data.top();
        auto res = LayerIO::Create().template Set<LayerIO>(TanhDerivative(tmp, tmp2));
        data.pop();
        return res;
    }
};

template <>
struct FeedbackOut_<false>
{
    template <typename ElementType, typename DeviceType>
    using InternalType = NullParameter;

    template <typename T, typename DataType>
    static void RecordData(T&& p_in, DataType&)
    {}

    template <typename TGrad, typename DataType>
    static auto FeedBack(const DataType&, const TGrad&)
    {
        return LayerIO::Create();
    }
};
}

template <typename TPolicies>
class TanhLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;

public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
    static constexpr bool IsUpdate = false;
    using InputType = LayerIO;
    using OutputType = LayerIO;

private:
    using ElementType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::ElementType;
    using DeviceType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::DeviceType;

private:
    using FeedbackOut_ = NSTanhLayer::FeedbackOut_<IsFeedbackOutput>;
    using InternalType = typename FeedbackOut_::template InternalType<ElementType, DeviceType>;

public:
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val = p_in.template Get<LayerIO>();

        using rawType = std::decay_t<decltype(val)>;
        static_assert(!std::is_same<rawType, NullParameter>::value, "parameter is invalid");

        auto tmp = Tanh(val);
        FeedbackOut_::RecordData(tmp, m_data);
        return LayerIO::Create().template Set<LayerIO>(tmp);
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        return FeedbackOut_::FeedBack(m_data, p_grad);
    }

    void NeutralInvariant()
    {
        LayerTraits::NeutralInvariant(m_data);
    }

private:
    InternalType m_data;
};
}
