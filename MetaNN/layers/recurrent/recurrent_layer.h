#pragma once

#include <MetaNN/layers/recurrent/gru_step.h>
#include <cassert>

namespace MetaNN
{
namespace NSRecurrentLayer
{
template <typename TStep, typename TPolicy> struct StepEnum2Type_;

template <typename TPolicy>
struct StepEnum2Type_<RecurrentLayerPolicy::StepTypeEnum::GRU, TPolicy>
{
    using type = GruStep<TPolicy>;
};

template <typename TStep, typename TPolicy>
using StepEnum2Type = typename StepEnum2Type_<TStep, TPolicy>::type;

enum class RnnStatus : char
{
    Forward,
    Backward
};

template <bool firstCall>
struct FeedForward_
{
    template <typename TIn, typename THid, typename TStep>
    static auto Eval(TIn&& in, THid& hidden, RnnStatus& status, TStep& step)
    {
        assert(status == NSRecurrentLayer::RnnStatus::Backward);
        status = NSRecurrentLayer::RnnStatus::Forward;
        auto res = step.FeedForward(std::forward<TIn>(in));

        hidden = MakeDynamic(res.template Get<LayerIO>());
        return res;
    }
};

template <>
struct FeedForward_<false>
{
    template <typename TIn, typename THid, typename TStep>
    static auto Eval(TIn in, THid& hidden, RnnStatus status, TStep& step)
    {
        assert(status == NSRecurrentLayer::RnnStatus::Forward);
        auto real_in = std::move(in).template Set<RnnLayerHiddenBefore>(hidden);
        auto res = step.FeedForward(std::move(real_in));
        hidden = MakeDynamic(res.template Get<LayerIO>());
        return res;
    }
};
}

template <typename TPolicies>
class RecurrentLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;

public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, TPolicies>::IsFeedbackOutput;
    static constexpr bool IsUpdate = PolicySelect<FeedbackPolicy, TPolicies>::IsUpdate;

private:
    static constexpr bool UseBptt = PolicySelect<RecurrentLayerPolicy, CurLayerPolicy>::UseBptt;

    using StepPolicy = typename std::conditional_t<(!IsFeedbackOutput) && IsUpdate && UseBptt,
                                                   ChangePolicy_<PFeedbackOutput, TPolicies>,
                                                   Identity_<TPolicies>>::type;

    using StepEnum = typename PolicySelect<RecurrentLayerPolicy, CurLayerPolicy>::StepType;
    using StepType = NSRecurrentLayer::StepEnum2Type<StepEnum, StepPolicy>;

    using ElementType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::ElementType;
    using DeviceType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::DeviceType;

    using DataType = DynamicData<ElementType, DeviceType, CategoryTags::Matrix>;

public:
    using InputType = typename StepType::InputType;
    using OutputType = typename StepType::OutputType;

public:
    template <typename...T>
    RecurrentLayer(T&&... params)
        : m_step(std::forward<T>(params)...)
        , m_status(NSRecurrentLayer::RnnStatus::Backward)
    {}

public:
    template <typename TInitializer, typename TLoad>
    void Init(TInitializer& initializer, TLoad& loader)
    {
        m_step.Init(initializer, loader);
    }

    template <typename TLoad>
    void LoadWeights(const TLoad& loader)
    {
        m_step.LoadWeights(loader);
    }

    template <typename TSave>
    void SaveWeights(TSave& saver)
    {
        m_step.SaveWeights(saver);
    }

    template <typename TGradCollector>
    void GradCollect(TGradCollector& col)
    {
        m_step.GradCollect(col);
    }

    template <typename TIn>
    auto FeedForward(TIn&& p_in)
    {
        auto& init = p_in.template Get<RnnLayerHiddenBefore>();
        using rawType = std::decay_t<decltype(init)>;
        constexpr static bool firstCall = !(std::is_same<rawType, NullParameter>::value);

        return NSRecurrentLayer::FeedForward_<firstCall>::Eval(std::forward<TIn>(p_in),
                                                               m_hiddens,
                                                               m_status,
                                                               m_step);
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        const bool addGrad = ((m_status == NSRecurrentLayer::RnnStatus::Backward) && (UseBptt));
        m_status = NSRecurrentLayer::RnnStatus::Backward;
        return m_step.FeedStepBackward(p_grad, m_hiddens, addGrad);
    }

    void NeutralInvariant()
    {
        m_status = NSRecurrentLayer::RnnStatus::Backward;
        m_hiddens = DataType();
        m_step.NeutralInvariant();
    }

private:
    StepType m_step;
    DataType m_hiddens;
    NSRecurrentLayer::RnnStatus m_status;
};
}
