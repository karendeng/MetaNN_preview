#pragma once
#include <MetaNN/facilities/named_params.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>

namespace MetaNN
{
namespace NSSoftmaxLayer
{
template <typename TVec, typename TIn,
          std::enable_if_t<std::is_same<TVec, InputPolicy::VectorTypeEnum::Row>::value>* = nullptr>
auto ForwardDispatch(const TIn& p_in)
{
    return RowSoftmax(p_in);
}

template <typename TVec, typename TIn,
          std::enable_if_t<std::is_same<TVec, InputPolicy::VectorTypeEnum::Column>::value>* = nullptr>
auto ForwardDispatch(const TIn& p_in)
{
    return ColSoftmax(p_in);
}

template <typename TVec, typename TGrad, typename TOut,
          std::enable_if_t<std::is_same<TVec, InputPolicy::VectorTypeEnum::Row>::value>* = nullptr>
auto BackwardDispatch(TGrad&& p_grad, TOut&& p_out)
{
    return RowSoftmaxDerivative(std::forward<TGrad>(p_grad), std::forward<TOut>(p_out));
}

template <typename TVec, typename TGrad, typename TOut,
          std::enable_if_t<std::is_same<TVec, InputPolicy::VectorTypeEnum::Column>::value>* = nullptr>
auto BackwardDispatch(TGrad&& p_grad, TOut&& p_out)
{
    return ColSoftmaxDerivative(std::forward<TGrad>(p_grad), std::forward<TOut>(p_out));
}

template <bool isFeedback>
struct FeedbackOut_
{
    template <typename ElementType, typename DeviceType>
    using InternalType = LayerTraits::LayerInternalBufType<ElementType, DeviceType,
                                                           CategoryTags::Matrix>;
                                                           
    template <typename T, typename DataType>
    static auto RecordData(const T& p_in, DataType& data)
    {
        auto res = MakeDynamic(p_in);
        data.push(res);
        return res;
    }

    template <typename TVec, typename TGrad, typename DataType>
    static auto FeedBack(DataType& data, const TGrad& grad)
    {
        if (data.empty())
        {
            throw std::runtime_error("Cannot do FeedBackward for Softmax Layer");
        }

        auto tmp = BackwardDispatch<TVec>(grad.template Get<LayerIO>(), data.top());
        auto res = LayerIO::Create().template Set<LayerIO>(tmp);
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
    static auto RecordData(T&& val, DataType&)
    {
        return std::forward<T>(val);
    }

    template <typename TVec, typename TGrad, typename DataType>
    static auto FeedBack(const DataType&, const TGrad&)
    {
        return LayerIO::Create();
    }
};
}

template <typename TPolicies>
class SoftmaxLayer
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
    using VectorType = typename PolicySelect<InputPolicy, CurLayerPolicy>::VectorType;

private:
    using FeedbackOut_ = NSSoftmaxLayer::FeedbackOut_<IsFeedbackOutput>;
    using DataType = typename FeedbackOut_::template InternalType<ElementType,
                                                                  DeviceType>;
public:
    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val = p_in.template Get<LayerIO>();

        using rawType = std::decay_t<decltype(val)>;
        static_assert(!std::is_same<rawType, NullParameter>::value, "parameter is invalid");

        auto tmp = NSSoftmaxLayer::ForwardDispatch<VectorType>(val);
        auto tmp2 = FeedbackOut_::RecordData(tmp, m_data);
        return LayerIO::Create().template Set<LayerIO>(tmp2);
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        return FeedbackOut_::template FeedBack<VectorType>(m_data, p_grad);
    }

    void NeutralInvariant()
    {
        LayerTraits::NeutralInvariant(m_data);
    }
private:
    DataType m_data;
};
}
