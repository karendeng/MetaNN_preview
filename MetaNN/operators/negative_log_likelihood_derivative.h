#pragma once

#include <MetaNN/operators/facilities/organizer.h>
#include <type_traits>
#include <vector>
#include <cmath>

namespace MetaNN
{
template <>
struct OperCategory_<TernaryOpTags::NegativeLogLikelihoodDerivative,
                     CategoryTags::Scalar, CategoryTags::Matrix, CategoryTags::Matrix>
{
    using type = CategoryTags::Matrix;
};

template <>
class OperOrganizer<TernaryOpTags::NegativeLogLikelihoodDerivative, CategoryTags::Matrix>
{
public:
    template <typename TD1, typename TD2, typename TD3>
    OperOrganizer(const TD1& data1, const TD2& data2, const TD3& data3)
        : m_rowNum(data2.RowNum())
        , m_colNum(data2.ColNum())
    {
        assert(data2.RowNum() == data3.RowNum());
        assert(data2.ColNum() == data3.ColNum());
    }

    size_t RowNum() const { return m_rowNum; }
    size_t ColNum() const { return m_colNum; }

private:
    size_t m_rowNum;
    size_t m_colNum;
};

template <typename TOp1, typename TOp2, typename TOp3>
struct OperElementType_<TernaryOpTags::NegativeLogLikelihoodDerivative,
                        TOp1, TOp2, TOp3>
{
    using type = typename TOp2::ElementType;
};

template <typename TOp1, typename TOp2, typename TOp3>
struct OperDeviceType_<TernaryOpTags::NegativeLogLikelihoodDerivative,
                        TOp1, TOp2, TOp3>
{
    using type = typename TOp2::DeviceType;
};

namespace NSNegativeLogLikelihoodDerivative
{
template <typename TElement, typename TOperHandle2, typename TOperHandle3,
          typename TDevice>
class EvalUnit;

template <typename TElement, typename TOperHandle2, typename TOperHandle3>
class EvalUnit<TElement, TOperHandle2, TOperHandle3, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
    using TOperandData = std::decay_t<decltype(std::declval<TOperHandle2>().Data())>;
public:
    using ElementType = typename TOperandData::ElementType;
    using DeviceType = typename TOperandData::DeviceType;
    static_assert(std::is_same<DeviceType, DeviceTags::CPU>::value,
                  "Device type mismatch");

    EvalUnit(TElement grad, TOperHandle2 operTar, TOperHandle3 operPre,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_grad(std::move(grad))
        , m_handleTar(std::move(operTar))
        , m_handlePre(std::move(operPre))
        , m_evalOutput(std::move(evalOutput)) {}

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_handleTar.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        it = depMap.find(m_handlePre.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }
    
    void Eval() override
    {
        const auto& p_tar = m_handleTar.Data();
        const auto& p_pre = m_handlePre.Data();

        const size_t rowNum = p_tar.RowNum();
        const size_t colNum = p_tar.ColNum();
        assert(p_pre.RowNum() == rowNum);
        assert(p_pre.ColNum() == colNum);
        
        m_evalOutput.Allocate(rowNum, colNum);
        auto& res = m_evalOutput.Data();
        
        assert(res.RowNum() == rowNum);
        assert(res.ColNum() == colNum);

        auto mem_v1 = LowerAccess(p_tar);
        auto mem_v2 = LowerAccess(p_pre);
        auto mem_res = LowerAccess(res);

        const size_t src1PackNum = mem_v1.RowLen();
        const size_t src2PackNum = mem_v2.RowLen();
        const size_t tgtPackNum = mem_res.RowLen();

        const ElementType* r1 = mem_v1.RawMemory();
        const ElementType* r2 = mem_v2.RawMemory();
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                r[j] = m_grad * (-r1[j] / r2[j]);
            }
            r1 += src1PackNum;
            r2 += src2PackNum;
            r += tgtPackNum;
        }
    }

private:
    TElement m_grad;
    TOperHandle2 m_handleTar;
    TOperHandle3 m_handlePre;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes,
              typename TOperator1, typename TOperator2, typename TOperator3>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1,
                             const TOperator2& oper2, const TOperator3& oper3)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        using RawEvalRes = typename TEvalRes::DataType;
        using ElementType = typename RawEvalRes::ElementType;
        using DeviceType = typename RawEvalRes::DeviceType;

        auto handle1 = oper2.EvalRegister();
        auto handle2 = oper3.EvalRegister();
        using UnitType = EvalUnit<ElementType, decltype(handle1), 
                                  decltype(handle2), DeviceType>;
        using GroupType = TrivalEvalGroup<UnitType>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        UnitType unit(oper1, std::move(handle1), std::move(handle2), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }
};
}

template <>
struct OperBuildInSeq_<TernaryOpTags::NegativeLogLikelihoodDerivative>
{
    using type = OperSeqContainer<NSNegativeLogLikelihoodDerivative::Calculator>;
};

template <typename TGrad, typename TP1, typename TP2>
struct OperNegativeLogLikelihoodDerivative_
{
// valid check
private:
    using rawGrad = std::decay_t<TGrad>;
    using rawM1 = std::decay_t<TP1>;
    using rawM2 = std::decay_t<TP2>;

public:
    static constexpr bool valid = IsScalar<rawGrad> && IsMatrix<rawM1> && IsMatrix<rawM2>;

public:
    static auto Eval(TGrad&& p_grad, TP1&& p_m1, TP2&& p_m2)
    {
        static_assert(std::is_same<typename rawM1::DeviceType, typename rawM2::DeviceType>::value,
                      "Matrices with different device types cannot do NegativeLogLikelihood derivative directly");
        static_assert(std::is_same<typename rawM1::ElementType, typename rawM2::ElementType>::value,
                      "Matrices with different element types cannot do NegativeLogLikelihood derivative directly");

        using ResType = TernaryOp<TernaryOpTags::NegativeLogLikelihoodDerivative,
                                  rawGrad, rawM1, rawM2>;
        return ResType(std::forward<TGrad>(p_grad), std::forward<TP1>(p_m1), std::forward<TP2>(p_m2));
    }
};

template <typename TGrad, typename TP1, typename TP2,
          std::enable_if_t<OperNegativeLogLikelihoodDerivative_<TGrad, TP1, TP2>::valid>* = nullptr>
auto NegativeLogLikelihoodDerivative(TGrad&& p_grad, TP1&& p_tar, TP2&& p_pre)
{
    return OperNegativeLogLikelihoodDerivative_<TGrad, TP1, TP2>
                ::Eval(std::forward<TGrad>(p_grad), std::forward<TP1>(p_tar), std::forward<TP2>(p_pre));
}
}
