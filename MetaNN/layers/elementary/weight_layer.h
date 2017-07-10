#pragma once
#include <MetaNN/facilities/named_params.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/policies/policy_operations.h>

namespace MetaNN
{
namespace NSWeightLayer
{
template <typename T> struct MatrixInfo;

template <>
struct MatrixInfo<InputPolicy::VectorTypeEnum::Row>
{
    template <typename ElementType, typename DeviceType>
    static auto AllocateMatrix(size_t inputLen, size_t outputLen)
    {
        return Matrix<ElementType, DeviceType>(inputLen, outputLen);
    }

    template <typename TM>
    static bool CheckMatrix(const TM& m, size_t inputLen, size_t outputLen)
    {
        return (m.RowNum() == inputLen) && (m.ColNum() == outputLen);
    }

    template <typename TIn, typename TWeight>
    static auto Eval(const TWeight& p_weight, const TIn& p_in)
    {
        return Dot(p_in, p_weight);
    }
};

template <>
struct MatrixInfo<InputPolicy::VectorTypeEnum::Column>
{
    template <typename ElementType, typename DeviceType>
    static auto AllocateMatrix(size_t inputLen, size_t outputLen)
    {
        return Matrix<ElementType, DeviceType>(outputLen, inputLen);
    }

    template <typename TM>
    static bool CheckMatrix(const TM& m, size_t inputLen, size_t outputLen)
    {
        return (m.RowNum() == outputLen) && (m.ColNum() == inputLen);
    }

    template <typename TIn, typename TWeight>
    static auto Eval(const TWeight& p_weight, const TIn& p_in)
    {
        return Dot(p_weight, p_in);
    }
};

template <bool update>
struct UpdateHelper
{
    template <typename ElementType, typename DeviceType>
    using DataType = LayerTraits::LayerInternalBufType<ElementType, DeviceType,
                                                       CategoryTags::Matrix>;

    template <typename TIn, typename UpdateInfo>
    static void Record(const TIn& p_in, UpdateInfo& info)
    {
        info.push(MakeDynamic(p_in));
    }

    template <typename TMatrixInfo, typename TGradIn, typename TData>
    static auto CalculateGrad(const TGradIn& p_grad, TData& update_info,
                              TData& gradInfo)
    {
        auto tmp = p_grad.template Get<LayerIO>();

        if (update_info.empty())
        {
            throw std::runtime_error("Cannot do FeedBackward for Weight Layer");
        }

        auto tw = Transpose(update_info.top());
        update_info.pop();
        auto res = TMatrixInfo::Eval(tmp, tw);

        gradInfo.push(MakeDynamic(res));
        return tmp;
    };

    template <typename TW, typename GradType, typename TGradCollector>
    static void GradCollect(const TW& weight,
                            GradType& grad,
                            TGradCollector& col)
    {
        LayerTraits::MatrixGradCollect(weight, grad, col);
    }
};

template <>
struct UpdateHelper<false>
{
    template <typename ElementType, typename DeviceType>
    using DataType = NullParameter;

    template <typename TIn, typename UpdateInfo>
    static void Record(TIn&&, UpdateInfo&) {}

    template <typename TMatrixInfo, typename TGradIn, typename TData>
    static auto CalculateGrad(const TGradIn& p_grad, TData&, TData&)
    {
        return p_grad.template Get<LayerIO>();
    }

    template <typename TW, typename GradType, typename TGradCollector>
    static void GradCollect(const TW&, const GradType&, TGradCollector&) { }
};

template <bool IsFeedbackOutput, typename MatrixInfo,
          typename TGrad, typename TWeight,
          std::enable_if_t<IsFeedbackOutput>* = nullptr>
static auto FeedBackHelper(const TGrad& p_grad, const TWeight& p_weight)
{
    auto tw = Transpose(p_weight);
    auto res = MatrixInfo::Eval(tw, p_grad);
    return LayerIO::Create().template Set<LayerIO>(std::move(res));
}

template <bool IsFeedbackOutput, typename MatrixInfo,
          typename TGrad, typename TWeight,
          std::enable_if_t<!IsFeedbackOutput>* = nullptr>
static auto FeedBackHelper(const TGrad&, const TWeight&)
{
    return LayerIO::Create();
}
}

template <typename TPolicies>
class WeightLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "TPolicies is not a policy container.");
    using CurLayerPolicy = PlainPolicy<TPolicies>;

public:
    static constexpr bool IsFeedbackOutput = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsFeedbackOutput;
    static constexpr bool IsUpdate = PolicySelect<FeedbackPolicy, CurLayerPolicy>::IsUpdate;
    using InputType = LayerIO;
    using OutputType = LayerIO;

private:
    using VectorType = typename PolicySelect<InputPolicy, CurLayerPolicy>::VectorType;
    using ElementType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::ElementType;
    using DeviceType = typename PolicySelect<OperandPolicy, CurLayerPolicy>::DeviceType;

public:
    WeightLayer(std::string p_name, size_t p_inLen, size_t p_outLen)
        : m_name(std::move(p_name))
        , m_inputLen(p_inLen)
        , m_outputLen(p_outLen)
    {
        if ((m_inputLen == 0) || (m_outputLen == 0))
        {
            throw std::runtime_error("Invalidate matrix size for weight layer");
        }
    }

private:
    using MatrixInfo = NSWeightLayer::MatrixInfo<VectorType>;
    using UpdateHelper = NSWeightLayer::UpdateHelper<IsUpdate>;
    using DataType = typename UpdateHelper::template DataType<ElementType, DeviceType>;

public:
    template <typename TInitializer, typename TLoad>
    void Init(TInitializer& initializer, TLoad& loader)
    {
        if (loader.find(m_name) != loader.end())
        {
            LoadWeights(loader);
            return;
        }
        m_weight = MatrixInfo::template AllocateMatrix<ElementType, DeviceType>(m_inputLen, m_outputLen);
        initializer.Eval(m_weight);
        loader[m_name] = m_weight;
    }

    template <typename TLoad>
    void LoadWeights(const TLoad& loader)
    {
        typename TLoad::const_iterator cit = loader.find(m_name);
        if (cit == loader.end())
        {
            throw std::runtime_error("Cannot find matrix with " + m_name);
        }

        const Matrix<ElementType, DeviceType>& m = cit->second;
        if (!MatrixInfo::CheckMatrix(m, m_inputLen, m_outputLen))
        {
            throw std::runtime_error("Load matrix error in WeightLayer");
        }

        m_weight = m;
    }

    template <typename TSave>
    void SaveWeights(TSave& saver) const
    {
        typename TSave::const_iterator cit = saver.find(m_name);
        if ((cit != saver.end()) && (cit->second != m_weight))
        {
            throw std::runtime_error("Duplicate save for matrix: " + m_name);
        }
        saver[m_name] = m_weight;
    }

    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val = p_in.template Get<LayerIO>();

        using rawType = std::decay_t<decltype(val)>;
        static_assert(!std::is_same<rawType, NullParameter>::value, "parameter is invalid");

        UpdateHelper::Record(val, m_updateInfo);
        auto res = MatrixInfo::Eval(m_weight, val);
        return LayerIO::Create().template Set<LayerIO>(std::move(res));
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        auto tmp = UpdateHelper::template CalculateGrad<MatrixInfo>(p_grad, m_updateInfo, m_gradInfo);
        return NSWeightLayer::FeedBackHelper<IsFeedbackOutput, MatrixInfo>(tmp, m_weight);
    }

    template <typename TGradCollector>
    void GradCollect(TGradCollector& col)
    {
        UpdateHelper::template GradCollect(m_weight, m_gradInfo, col);
    }

    void NeutralInvariant()
    {
        LayerTraits::NeutralInvariant(m_updateInfo);
        LayerTraits::NeutralInvariant(m_gradInfo);
    }

private:
    const std::string m_name;
    const size_t m_inputLen;
    const size_t m_outputLen;

    Matrix<ElementType, DeviceType> m_weight;
    DataType m_updateInfo;
    DataType m_gradInfo;
};
}
