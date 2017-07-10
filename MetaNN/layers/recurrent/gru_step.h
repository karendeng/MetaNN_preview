#pragma once

#include <MetaNN/layers/compose/compose_kernel.h>
#include <MetaNN/layers/elementary/weight_layer.h>
#include <MetaNN/layers/elementary/add_layer.h>
#include <MetaNN/layers/elementary/sigmoid_layer.h>
#include <MetaNN/layers/elementary/tanh_layer.h>
#include <MetaNN/layers/elementary/element_mul_layer.h>
#include <MetaNN/layers/elementary/interpolate_layer.h>
#include <MetaNN/layers/facilities/common_io.h>

namespace MetaNN
{
struct GruInput
    : public NamedParams<RnnLayerHiddenBefore,
                         struct GruInputBelow,
                         struct GruUpdateBelow,
                         struct GruResetBelow> {};

template <typename TPolicies> class GruStep;

template <>
struct Sublayerof<GruStep>
{
    struct UpdateWeight; struct UpdateAdd; struct UpdateAct;
    struct ResetWeight; struct ResetAdd; struct ResetAct;
    struct InputWeight; struct InputAdd; struct InputAct;
    struct ElemMul; struct Interpolate;
};

namespace NSGruStep
{
using UpdateWeight = typename Sublayerof<GruStep>::UpdateWeight;
using UpdateAdd = typename Sublayerof<GruStep>::UpdateAdd;
using UpdateAct = typename Sublayerof<GruStep>::UpdateAct;
using ResetWeight = typename Sublayerof<GruStep>::ResetWeight;
using ResetAdd = typename Sublayerof<GruStep>::ResetAdd;
using ResetAct = typename Sublayerof<GruStep>::ResetAct;
using InputWeight = typename Sublayerof<GruStep>::InputWeight;
using InputAdd = typename Sublayerof<GruStep>::InputAdd;
using InputAct = typename Sublayerof<GruStep>::InputAct;
using ElemMul = typename Sublayerof<GruStep>::ElemMul;
using Interpolate = typename Sublayerof<GruStep>::Interpolate;

using Topology = ComposeTopology<
        SubLayer<UpdateWeight, WeightLayer>,
        SubLayer<UpdateAdd, AddLayer>,
        SubLayer<UpdateAct, SigmoidLayer>,
        InConnect<RnnLayerHiddenBefore, UpdateWeight, LayerIO>,
        InConnect<GruUpdateBelow, UpdateAdd, AddLayerIn1>,
        InternalConnect<UpdateWeight, LayerIO, UpdateAdd, AddLayerIn2>,
        InternalConnect<UpdateAdd, LayerIO, UpdateAct, LayerIO>,

        SubLayer<ResetWeight, WeightLayer>,
        SubLayer<ResetAdd, AddLayer>,
        SubLayer<ResetAct, SigmoidLayer>,
        InConnect<RnnLayerHiddenBefore, ResetWeight, LayerIO>,
        InConnect<GruResetBelow, ResetAdd, AddLayerIn1>,
        InternalConnect<ResetWeight, LayerIO, ResetAdd, AddLayerIn2>,
        InternalConnect<ResetAdd, LayerIO, ResetAct, LayerIO>,

        SubLayer<ElemMul, ElementMulLayer>,
        InConnect<RnnLayerHiddenBefore, ElemMul, ElementMulLayerIn1>,
        InternalConnect<ResetAct, LayerIO, ElemMul, ElementMulLayerIn2>,

        SubLayer<InputWeight, WeightLayer>,
        SubLayer<InputAdd, AddLayer>,
        SubLayer<InputAct, TanhLayer>,
        InternalConnect<ElemMul, LayerIO, InputWeight, LayerIO>,
        InConnect<GruInputBelow, InputAdd, AddLayerIn1>,
        InternalConnect<InputWeight, LayerIO, InputAdd, AddLayerIn2>,
        InternalConnect<InputAdd, LayerIO, InputAct, LayerIO>,

        SubLayer<Interpolate, InterpolateLayer>,
        InternalConnect<InputAct, LayerIO, Interpolate, InterpolateLayerWeight1>,
        InConnect<RnnLayerHiddenBefore, Interpolate, InterpolateLayerWeight2>,
        InternalConnect<UpdateAct, LayerIO, Interpolate, InterpolateLayerLambda>,

        OutConnect<Interpolate, LayerIO, LayerIO>>;

template <typename TPolicies>
using Base = ComposeKernel<GruInput, LayerIO, TPolicies, Topology>;
}

template <typename TPolicies>
class GruStep : public NSGruStep::Base<TPolicies>
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");

    using TBase = typename NSGruStep::template Base<TPolicies>;
    using TupleProcessor = typename TBase::TupleProcessor;

public:
    GruStep(const std::string& p_name, size_t p_inLen, size_t p_outLen)
        : TBase(TupleProcessor::Create()
                        .template Set<NSGruStep::UpdateWeight>(p_name + "-update", p_inLen, p_outLen)
                        .template Set<NSGruStep::UpdateAdd>()
                        .template Set<NSGruStep::UpdateAct>()
                        .template Set<NSGruStep::ResetWeight>(p_name + "-reset", p_inLen, p_outLen)
                        .template Set<NSGruStep::ResetAdd>()
                        .template Set<NSGruStep::ResetAct>()
                        .template Set<NSGruStep::InputWeight>(p_name + "-input", p_inLen, p_outLen)
                        .template Set<NSGruStep::InputAdd>()
                        .template Set<NSGruStep::InputAct>()
                        .template Set<NSGruStep::ElemMul>()
                        .template Set<NSGruStep::Interpolate>()) {}

public:
    template <typename TGrad, typename THid>
    auto FeedStepBackward(TGrad&& p_grad, THid& hiddens, bool addGrad)
    {
        if (addGrad)
        {
            auto gradVal = p_grad.template Get<LayerIO>();
            auto res = TBase::FeedBackward(LayerIO::Create().template Set<LayerIO>(gradVal + hiddens));
            hiddens = MakeDynamic(res.template Get<RnnLayerHiddenBefore>());

            return GruInput::Create()
                    .template Set<GruInputBelow>(MakeDynamic(res.template Get<GruInputBelow>()))
                    .template Set<GruUpdateBelow>(MakeDynamic(res.template Get<GruUpdateBelow>()))
                    .template Set<GruResetBelow>(MakeDynamic(res.template Get<GruResetBelow>()));
        }
        else
        {
            auto res = TBase::FeedBackward(std::forward<TGrad>(p_grad));
            hiddens = MakeDynamic(res.template Get<RnnLayerHiddenBefore>());

            return GruInput::Create()
                    .template Set<GruInputBelow>(MakeDynamic(res.template Get<GruInputBelow>()))
                    .template Set<GruUpdateBelow>(MakeDynamic(res.template Get<GruUpdateBelow>()))
                    .template Set<GruResetBelow>(MakeDynamic(res.template Get<GruResetBelow>()));
        }
    }
};
}
