#pragma once
#include <MetaNN/data/batch/general.h>
#include <MetaNN/data/dynamic.h>
#include <MetaNN/facilities/named_params.h>
#include <MetaNN/layers/facilities/common_io.h>
#include <MetaNN/layers/facilities/policies.h>
#include <MetaNN/layers/facilities/traits.h>
#include <MetaNN/policies/policy_operations.h>
#include <MetaNN/operators/collapse.h>

namespace MetaNN
{
namespace NSBiasLayer
{
template <typename TVectorType,
          std::enable_if_t<std::is_same<TVectorType, InputPolicy::VectorTypeEnum::Row>::value>* = nullptr>
void InitBiasVector(size_t len, size_t& row, size_t& col)
{
    row = 1;
    col = len;
}

template <typename TVectorType,
          std::enable_if_t<std::is_same<TVectorType, InputPolicy::VectorTypeEnum::Column>::value>* = nullptr>
void InitBiasVector(size_t len, size_t& row, size_t& col)
{
    row = len;
    col = 1;
}

template <bool isUpdate>
struct Update_
{
    template <typename ElementType, typename DeviceType>
    using DataType = LayerTraits::LayerInternalBufType<ElementType, DeviceType,
                                                       CategoryTags::Matrix>;

    template <typename TGrad, typename TBias>
    static void CheckGrad(const TGrad& grad, const TBias& bias)
    {
        const auto& tmp = grad.template Get<LayerIO>();
        assert((tmp.RowNum() == bias.RowNum()) && (tmp.ColNum() == bias.ColNum()));
    }

    template <typename TIn, typename TGrad>
    static void RecordGrad(const TIn& p_in, TGrad& g)
    {
        g.push(MakeDynamic(p_in.template Get<LayerIO>()));
    }

    template <typename TWeight, typename TGrad, typename TGradCollector>
    static void GradCollect(const TWeight& weight,
                            TGrad& grad,
                            TGradCollector& col)
    {
        LayerTraits::MatrixGradCollect(weight, grad, col);
    }
};

template <>
struct Update_<false>
{
    template <typename ElementType, typename DeviceType>
    using DataType = NullParameter;

    template <typename TGrad, typename TBias>
    static void CheckGrad(const TGrad&, const TBias&) {}

    template <typename TIn, typename TGrad>
    static void RecordGrad(const TIn& p_in, TGrad&)
    {}

    template <typename TWeight, typename TGrad, typename TGradCollector>
    static void GradCollect(const TWeight&,
                            const TGrad&, TGradCollector&) {}
};
}

template <typename TPolicies>
class BiasLayer
{
    static_assert(IsPolicyContainer<TPolicies>, "Parameter is not policy container.");
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
    BiasLayer(std::string p_name, size_t p_vecLen)
        : m_name(std::move(p_name))
    {
        if (p_vecLen == 0)
        {
            throw std::runtime_error("Invalidate input len for bias layer");
        }
        NSBiasLayer::InitBiasVector<VectorType>(p_vecLen, m_rowNum, m_colNum);
    }
    
    BiasLayer(std::string p_name, size_t p_rowNum, size_t p_colNum)
        : m_name(std::move(p_name))
        , m_rowNum(p_rowNum)
        , m_colNum(p_colNum)
    {
        if ((m_rowNum == 0) || (m_colNum == 0))
        {
            throw std::runtime_error("Invalidate row/col num for bias layer.");
        }
    }

private:
    using Update_ = NSBiasLayer::Update_<IsUpdate>;
    using DataType = typename Update_::template DataType<ElementType, DeviceType>;

public:
    template <typename TInitializer, typename TLoad>
    void Init(TInitializer& initializer, TLoad& loader)
    {
        if (loader.find(m_name) != loader.end())
        {
            LoadWeights(loader);
            return;
        }

        m_bias = Matrix<ElementType, DeviceType>(m_rowNum, m_colNum);
        initializer.Eval(m_bias);
        loader[m_name] = m_bias;
    }

    template <typename TLoad>
    void LoadWeights(const TLoad& loader)
    {
        auto cit = loader.find(m_name);
        if (cit == loader.end())
        {
            throw std::runtime_error("Cannot find matrix with " + m_name);
        }

        const Matrix<ElementType, DeviceType>& m = cit->second;
        if ((m.RowNum() != m_rowNum) || (m.ColNum() != m_colNum))
        {
            throw std::runtime_error("Load matrix error in BiasLayer");
        }

        m_bias = m;
    }

    template <typename TSave>
    void SaveWeights(TSave& saver) const
    {
        auto cit = saver.find(m_name);
        if ((cit != saver.end()) && (cit->second != m_bias))
        {
            throw std::runtime_error("Duplicate save for matrix: " + m_name);
        }
        saver[m_name] = m_bias;
    }

    template <typename TIn>
    auto FeedForward(const TIn& p_in)
    {
        const auto& val = p_in.template Get<LayerIO>();
        return LayerIO::Create().template Set<LayerIO>(val + m_bias);
    }

    template <typename TGrad>
    auto FeedBackward(const TGrad& p_grad)
    {
        Update_::CheckGrad(p_grad, m_bias);
        Update_::RecordGrad(p_grad, m_grad);
        return p_grad;
    }

    template <typename TGradCollector>
    void GradCollect(TGradCollector& col)
    {
        Update_::template GradCollect(m_bias, m_grad, col);
    }

    void NeutralInvariant()
    {
        LayerTraits::NeutralInvariant(m_grad);
    }

private:
    const std::string m_name;
    size_t m_rowNum;
    size_t m_colNum;

    Matrix<ElementType, DeviceType> m_bias;
    DataType m_grad;
};
}
