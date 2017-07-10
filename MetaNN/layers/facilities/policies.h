#pragma once

#include <MetaNN/policies/policy_macro_begin.h>
#include <MetaNN/data/facilities/tags.h>
namespace MetaNN
{
struct FeedbackPolicy
{
    struct IsUpdateValue;
    struct IsFeedbackOutputValue;

    static constexpr bool IsUpdate = false;
    static constexpr bool IsFeedbackOutput = false;
};
ValuePolicyObj(PUpdate,   FeedbackPolicy, IsUpdate, true);
ValuePolicyObj(PNoUpdate, FeedbackPolicy, IsUpdate, false);
ValuePolicyObj(PFeedbackOutput,   FeedbackPolicy, IsFeedbackOutput, true);
ValuePolicyObj(PFeedbackNoOutput, FeedbackPolicy, IsFeedbackOutput, false);

struct InputPolicy
{
    struct VectorTypeEnum
    {
        struct Row;
        struct Column;
    };

    using VectorType = VectorTypeEnum::Column;
};
EnumPolicyObj(PRowVecInput, InputPolicy, VectorType, Row);
EnumPolicyObj(PColVecInput, InputPolicy, VectorType, Column);

struct OperandPolicy
{
    struct ElementTypeType;
    struct DeviceTypeEnum : public MetaNN::DeviceTags {};

    using ElementType = float;
    using DeviceType = DeviceTypeEnum::CPU;
};
TypePolicyObj(PFloatElement,  OperandPolicy, ElementType, float);
EnumPolicyObj(PCPUDevice,    OperandPolicy, DeviceType, CPU);

struct SingleLayerPolicy
{
    struct ActionTypeEnum
    {
        struct Sigmoid;
        struct Tanh;
    };
    struct HasBiasValue;

    using ActionType = ActionTypeEnum::Sigmoid;
    static constexpr bool HasBias = true;
};
EnumPolicyObj(PSigmoidAction, SingleLayerPolicy, ActionType, Sigmoid);
EnumPolicyObj(PTanhAction, SingleLayerPolicy, ActionType, Tanh);
ValuePolicyObj(PBiasSingleLayer,  SingleLayerPolicy, HasBias, true);
ValuePolicyObj(PNoBiasSingleLayer, SingleLayerPolicy, HasBias, false);

struct RecurrentLayerPolicy
{
    struct StepTypeEnum
    {
        struct GRU;
    };
    struct UseBpttValue;

    using StepType = StepTypeEnum::GRU;
    constexpr static bool UseBptt = true;
};
EnumPolicyObj(PRecGRUStep, RecurrentLayerPolicy, StepType, GRU);
ValuePolicyObj(PEnableBptt,  RecurrentLayerPolicy, UseBptt, true);
ValuePolicyObj(PDisableBptt,  RecurrentLayerPolicy, UseBptt, false);
}
#include <MetaNN/policies/policy_macro_end.h>
