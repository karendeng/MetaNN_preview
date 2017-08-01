#pragma once

#include <MetaNN/operators/operators.h>

namespace MetaNN
{
namespace NSRowSoftmaxDerivative
{
namespace CaseNLL
{
template <typename T1, typename T2>
constexpr bool Valid = false;

template <typename T1, typename T2, typename T3>
constexpr bool Valid<TernaryOp<TernaryOpTags::NegativeLogLikelihoodDerivative,
                               T1, T2, T3>,
                     T3> = true;
                     
template <typename TOperHandle1, typename TOperHandle2, typename TElem, typename TDevice>
class Case1EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TElem>
class Case1EvalUnit<TOperHandle1, TOperHandle2, TElem, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    Case1EvalUnit(TOperHandle1 grad,
                  size_t hotPos,
                  TOperHandle2 handlePre,
                  EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_grad(std::move(grad))
        , m_hotPos(hotPos)
        , m_handlePre(std::move(handlePre))
        , m_evalOutput(std::move(evalOutput)) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_handlePre.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));
        
        it = depMap.find(m_grad.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }

    void Eval() override
    {
        const auto& grad = m_grad.Data().Value();
        const auto& p_pre = m_handlePre.Data();
        assert(p_pre.RowNum() == 1);
        
        size_t colNum = p_pre.ColNum();
        assert(m_hotPos < colNum);
        
        m_evalOutput.Allocate(1, colNum);
        auto& res = m_evalOutput.MutableData();

        auto mem_res = LowerAccess(res);
        ElementType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < colNum; ++i)
        {
            r[i] = p_pre(0, i) * grad;
        }
        r[m_hotPos] -= grad;
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_grad;
    size_t m_hotPos;
    TOperHandle2 m_handlePre;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3,
          typename TElem, typename TDevice>
class Case2EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3, typename TElem>
class Case2EvalUnit<TOperHandle1, TOperHandle2, TOperHandle3, TElem, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    Case2EvalUnit(TOperHandle1 grad,
                  TOperHandle2 handleTar,
                  TOperHandle3 handlePre,
                  EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_grad(grad)
        , m_handleTar(std::move(handleTar))
        , m_handlePre(std::move(handlePre))
        , m_evalOutput(std::move(evalOutput)) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_handleTar.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));
        
        it = depMap.find(m_grad.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        it = depMap.find(m_handlePre.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }
    
    void Eval() override
    {
        const auto& grad = m_grad.Data().Value();
        const auto& p_tar = m_handleTar.Data();
        const auto& p_pre = m_handlePre.Data();

        assert(p_tar.RowNum() == 1);
        assert(p_pre.RowNum() == 1);

        size_t colNum = p_tar.ColNum();
        assert(colNum == p_pre.ColNum());
        
        m_evalOutput.Allocate(1, colNum);
        auto& res = m_evalOutput.MutableData();

        using StorageType = typename Scalar<ElementType, DeviceType>::StorageType;
        StorageType sum = StorageType();
        for (size_t i = 0; i < colNum; ++i)
        {
            sum += p_tar(0, i);
        }

        auto mem_res = LowerAccess(res);
        StorageType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < colNum; ++i)
        {
            r[i] = (p_pre(0, i) * sum - p_tar(0, i)) * grad;
        }
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_grad;
    TOperHandle2 m_handleTar;
    TOperHandle3 m_handlePre;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseRem, typename TEvalRes, typename TOperator1, typename TOperator2,
              std::enable_if_t<!Valid<TOperator1, TOperator2>>* = nullptr>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        using THead = OperSeqHead<TCaseRem>;
        using TTail = OperSeqTail<TCaseRem>;
        THead::template EvalRegister<TTail>(evalRes, oper1, oper2);
    }
    
    template <typename TCaseRem, typename TEvalRes, typename TOperator1, typename TOperator2,
              std::enable_if_t<Valid<TOperator1, TOperator2>>* = nullptr>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        const auto& softmax_res = oper1.Operand3();
        if (softmax_res != oper2)
        {
            using THead = OperSeqHead<TCaseRem>;
            using TTail = OperSeqTail<TCaseRem>;
            THead::template EvalRegister<TTail>(evalRes, oper1, oper2);
            return;
        }
        
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();

        if (auto ptr = oper1.Operand2().template TypeCast<OneHotRowVector<ElementType, DeviceType>>())
        {
            auto handle1 = oper1.Operand1().EvalRegister();
            auto handle2 = softmax_res.EvalRegister();
            using EvalUnit = Case1EvalUnit<decltype(handle1), decltype(handle2), ElementType, DeviceType>;
            using GroupType = TrivalEvalGroup<EvalUnit>;

            EvalUnit unit(std::move(handle1), ptr->HotPos(), std::move(handle2), std::move(outHandle));
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
            return;
        }
        else
        {
            auto handle1 = oper1.Operand1().EvalRegister();
            auto handle2 = oper1.Operand2().EvalRegister();
            auto handle3 = softmax_res.EvalRegister();

            using EvalUnit = Case2EvalUnit<decltype(handle1), decltype(handle2), decltype(handle3),
                                           ElementType, DeviceType>;
            using GroupType = TrivalEvalGroup<EvalUnit>;

            EvalUnit unit(std::move(handle1), std::move(handle2), std::move(handle3), std::move(outHandle));
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
        }
    }
};
}

namespace CaseGen
{
template <typename TOperHandle1, typename TOperHandle2, typename TElem, typename TDevice>
class EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TElem>
class EvalUnit<TOperHandle1, TOperHandle2, TElem, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    EvalUnit(TOperHandle1 oper1,
             TOperHandle2 oper2,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_oper1(std::move(oper1))
        , m_oper2(std::move(oper2))
        , m_evalOutput(evalOutput) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_oper1.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        it = depMap.find(m_oper2.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }
    
    void Eval() override
    {
        const auto& p_grad = m_oper1.Data();
        const auto& p_sout = m_oper2.Data();
        size_t colNum = p_grad.ColNum();
        assert(p_grad.RowNum() == 1);
        assert(p_sout.RowNum() == 1);
        assert(colNum == p_sout.ColNum());
                
        Matrix<ElementType, DeviceType> tmp(colNum, colNum);
        for (size_t i = 0; i < colNum; ++i)
        {
            for (size_t j = 0; j < colNum; ++j)
            {
                tmp.SetValue(i, j, -1 * p_sout(0, i) * p_sout(0, j));
            }
            tmp.SetValue(i, i, p_sout(0, i) + tmp(i, i));
        }

        auto tempHandle = tmp.EvalRegister();
        using EvalUnit = NSDot::NSCaseGen::EvalUnit<decltype(m_oper1), decltype(tempHandle), ElementType, DeviceType>;
        using GroupType = TrivalEvalGroup<EvalUnit>;

        const void* dataPtr = m_evalOutput.DataPtr();
        EvalUnit unit(m_oper1, std::move(tempHandle), std::move(m_evalOutput));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }

private:
    TOperHandle1 m_oper1;
    TOperHandle2 m_oper2;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOperator1, typename TOperator2>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        auto handle1 = oper1.EvalRegister();
        auto handle2 = oper2.EvalRegister();

        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;

        using EvalUnit = EvalUnit<decltype(handle1), decltype(handle2), ElementType, DeviceType>;
        using GroupType = TrivalEvalGroup<EvalUnit>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        EvalUnit unit(std::move(handle1), std::move(handle2), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }
};
}
}

template <>
struct OperBuildInSeq_<BinaryOpTags::RowSoftmaxDerivative>
{
    using type = OperSeqContainer<NSRowSoftmaxDerivative::CaseNLL::Calculator,
                                  NSRowSoftmaxDerivative::CaseGen::Calculator>;
};

template <typename TGrad, typename TSOut>
struct OperRowSoftmaxDerivative_
{
// valid check
private:
    using rawGrad = RemConstRef<TGrad>;
    using rawSOut = RemConstRef<TSOut>;

public:
    static constexpr bool valid = (IsMatrix<rawGrad> && IsMatrix<rawSOut>);

public:
    static auto Eval(TGrad&& p_grad, TSOut&& p_sout)
    {
        static_assert(std::is_same<typename rawGrad::ElementType, typename rawSOut::ElementType>::value,
                      "Element type mismatch.");
        static_assert(std::is_same<typename rawGrad::DeviceType, typename rawSOut::DeviceType>::value,
                      "Device type mismatch.");

        using ResType = BinaryOp<BinaryOpTags::RowSoftmaxDerivative, rawGrad, rawSOut>;
        return ResType(std::forward<TGrad>(p_grad), std::forward<TSOut>(p_sout));
    }
};

template <typename TGrad, typename TSOut,
          std::enable_if_t<OperRowSoftmaxDerivative_<TGrad, TSOut>::valid>* = nullptr>
auto RowSoftmaxDerivative(TGrad&& p_grad, TSOut&& p_sout)
{
    return OperRowSoftmaxDerivative_<TGrad, TSOut>::Eval(std::forward<TGrad>(p_grad),
                                                         std::forward<TSOut>(p_sout));
}

namespace NSColSoftmaxDerivative
{
namespace CaseNLL
{
template <typename T1, typename T2>
constexpr bool Valid = false;

template <typename T1, typename T2, typename T3>
constexpr bool Valid<TernaryOp<TernaryOpTags::NegativeLogLikelihoodDerivative,
                               T1, T2, T3>,
                     T3> = true;

template <typename TOperHandle1, typename TOperHandle2, typename TElem, typename TDevice>
class Case1EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TElem>
class Case1EvalUnit<TOperHandle1, TOperHandle2, TElem, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    Case1EvalUnit(TOperHandle1 grad,
                  size_t hotPos,
                  TOperHandle2 handlePre,
                  EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_grad(std::move(grad))
        , m_hotPos(hotPos)
        , m_handlePre(std::move(handlePre))
        , m_evalOutput(std::move(evalOutput)) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_handlePre.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));
        
        it = depMap.find(m_grad.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }

    void Eval() override
    {
        const auto& grad = m_grad.Data().Value();
        const auto& p_pre = m_handlePre.Data();
        assert(p_pre.ColNum() == 1);

        size_t rowNum = p_pre.RowNum();
        assert(m_hotPos < rowNum);
        
        m_evalOutput.Allocate(rowNum, 1);
        auto& res = m_evalOutput.MutableData();

        auto mem_res = LowerAccess(res);
        using StorageType = typename Scalar<ElementType, DeviceType>::StorageType;
        StorageType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            r[i] = p_pre(i, 0) * grad;
        }
        r[m_hotPos] -= grad;
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_grad;
    size_t m_hotPos;
    TOperHandle2 m_handlePre;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3,
          typename TElem, typename TDevice>
class Case2EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TOperHandle3, typename TElem>
class Case2EvalUnit<TOperHandle1, TOperHandle2, TOperHandle3, TElem, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = typename DeviceTags::CPU;

    Case2EvalUnit(TOperHandle1 grad,
                  TOperHandle2 handleTar,
                  TOperHandle3 handlePre,
                  EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_grad(grad)
        , m_handleTar(std::move(handleTar))
        , m_handlePre(std::move(handlePre))
        , m_evalOutput(std::move(evalOutput)) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_handleTar.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));
        
        it = depMap.find(m_grad.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        it = depMap.find(m_handlePre.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }
    
    void Eval() override
    {
        const auto& grad = m_grad.Data().Value();
        const auto& p_tar = m_handleTar.Data();
        const auto& p_pre = m_handlePre.Data();

        assert(p_tar.ColNum() == 1);
        assert(p_pre.ColNum() == 1);

        size_t rowNum = p_tar.RowNum();
        assert(rowNum == p_pre.RowNum());
        
        m_evalOutput.Allocate(rowNum, 1);
        auto& res = m_evalOutput.MutableData();

        using StorageType = typename Scalar<ElementType, DeviceType>::StorageType;
        StorageType sum = StorageType();
        for (size_t i = 0; i < rowNum; ++i)
        {
            sum += p_tar(i, 0);
        }

        auto mem_res = LowerAccess(res);
        StorageType* r = mem_res.MutableRawMemory();

        for (size_t i = 0; i < rowNum; ++i)
        {
            r[i] = (p_pre(i, 0) * sum - p_tar(i, 0)) * grad;
        }
        m_evalOutput.SetEval();
    }

private:
    TOperHandle1 m_grad;
    TOperHandle2 m_handleTar;
    TOperHandle3 m_handlePre;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseRem, typename TEvalRes, typename TOperator1, typename TOperator2,
              std::enable_if_t<!Valid<TOperator1, TOperator2>>* = nullptr>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        using THead = OperSeqHead<TCaseRem>;
        using TTail = OperSeqTail<TCaseRem>;
        THead::template EvalRegister<TTail>(evalRes, oper1, oper2);
    }
    
    template <typename TCaseRem, typename TEvalRes, typename TOperator1, typename TOperator2,
              std::enable_if_t<Valid<TOperator1, TOperator2>>* = nullptr>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        const auto& softmax_res = oper1.Operand3();
        if (softmax_res != oper2)
        {
            using THead = OperSeqHead<TCaseRem>;
            using TTail = OperSeqTail<TCaseRem>;
            THead::template EvalRegister<TTail>(evalRes, oper1, oper2);
            return;
        }
        
        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;
        
        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        
        if (auto ptr = oper1.Operand2().template TypeCast<OneHotColVector<ElementType, DeviceType>>())
        {
            auto handle1 = oper1.Operand1().EvalRegister();
            auto handle2 = softmax_res.EvalRegister();
            using EvalUnit = Case1EvalUnit<decltype(handle1), decltype(handle2), ElementType, DeviceType>;
            using GroupType = TrivalEvalGroup<EvalUnit>;

            EvalUnit unit(std::move(handle1), ptr->HotPos(), std::move(handle2), std::move(outHandle));
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
            return;
        }
        else
        {
            auto handle1 = oper1.Operand1().EvalRegister();
            auto handle2 = oper1.Operand2().EvalRegister();
            auto handle3 = softmax_res.EvalRegister();

            using EvalUnit = Case2EvalUnit<decltype(handle1), decltype(handle2), decltype(handle3),
                                           ElementType, DeviceType>;
            using GroupType = TrivalEvalGroup<EvalUnit>;

            EvalUnit unit(std::move(handle1), std::move(handle2), std::move(handle3), std::move(outHandle));
            EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
        }
    }
};
} // namespace CaseNLL

namespace CaseGen
{
template <typename TOperHandle1, typename TOperHandle2, typename TElem, typename TDevice>
class EvalUnit;

template <typename TOperHandle1, typename TOperHandle2, typename TElem>
class EvalUnit<TOperHandle1, TOperHandle2, TElem, DeviceTags::CPU>
    : public BaseEvalUnit<DeviceTags::CPU>
{
public:
    using ElementType = TElem;
    using DeviceType = DeviceTags::CPU;

    EvalUnit(TOperHandle1 oper1,
             TOperHandle2 oper2,
             EvalHandle<Matrix<ElementType, DeviceType>> evalOutput)
        : m_oper1(std::move(oper1))
        , m_oper2(std::move(oper2))
        , m_evalOutput(evalOutput) { }

    size_t OperandDepth(const std::unordered_map<const void*, size_t>& depMap) const
    {
        int res = -1;

        auto it = depMap.find(m_oper1.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        it = depMap.find(m_oper2.DataPtr());
        if (it != depMap.end()) res = std::max(res, (int)(it->second));

        return (size_t)res;
    }
    
    void Eval() override
    {
        const auto& p_grad = m_oper1.Data();
        const auto& p_sout = m_oper2.Data();

        assert(p_grad.ColNum() == 1);
        assert(p_sout.ColNum() == 1);

        size_t rowNum = p_grad.RowNum();
        assert(rowNum == p_sout.RowNum());

        Matrix<ElementType, DeviceType> tmp(rowNum, rowNum);
        for (size_t i = 0; i < rowNum; ++i)
        {
            for (size_t j = 0; j < rowNum; ++j)
            {
                tmp.SetValue(i, j, -1 * p_sout(i, 0) * p_sout(j, 0));
            }
            tmp.SetValue(i, i, p_sout(i, 0) + tmp(i, i));
        }

        auto tempHandle = tmp.EvalRegister();
        using EvalUnit = NSDot::NSCaseGen::EvalUnit<decltype(tempHandle), decltype(m_oper1), ElementType, DeviceType>;
        using GroupType = TrivalEvalGroup<EvalUnit>;

        const void* dataPtr = m_evalOutput.DataPtr();
        EvalUnit unit(std::move(tempHandle), m_oper1, std::move(m_evalOutput));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }

private:
    TOperHandle1 m_oper1;
    TOperHandle2 m_oper2;
    EvalHandle<Matrix<ElementType, DeviceType>> m_evalOutput;
};

struct Calculator
{
    template <typename TCaseTail, typename TEvalRes, typename TOperator1, typename TOperator2>
    static void EvalRegister(TEvalRes& evalRes, const TOperator1& oper1, const TOperator2& oper2)
    {
        static_assert(std::is_same<TCaseTail, OperSeqContainer<>>::value,
                      "General Case is not the last one");
                      
        auto handle1 = oper1.EvalRegister();
        auto handle2 = oper2.EvalRegister();

        using ElementType = typename TEvalRes::DataType::ElementType;
        using DeviceType = typename TEvalRes::DataType::DeviceType;

        using EvalUnit = EvalUnit<decltype(handle1), decltype(handle2), ElementType, DeviceType>;
        using GroupType = TrivalEvalGroup<EvalUnit>;

        auto outHandle = evalRes.Handle();
        const void* dataPtr = outHandle.DataPtr();
        EvalUnit unit(std::move(handle1), std::move(handle2), std::move(outHandle));
        EvalPlan<DeviceType>::template Register<GroupType>(std::move(unit), dataPtr);
    }
};
} // namespace CaseNLL
}

template <>
struct OperBuildInSeq_<BinaryOpTags::ColSoftmaxDerivative>
{
    using type = OperSeqContainer<NSColSoftmaxDerivative::CaseNLL::Calculator,
                                  NSColSoftmaxDerivative::CaseGen::Calculator>;
};

template <typename TGrad, typename TSOut>
struct OperColSoftmaxDerivative_
{
// valid check
private:
    using rawGrad = RemConstRef<TGrad>;
    using rawSOut = RemConstRef<TSOut>;

public:
    static constexpr bool valid = (IsMatrix<rawGrad> && IsMatrix<rawSOut>);

public:
    static auto Eval(TGrad&& p_grad, TSOut&& p_sout)
    {
        static_assert(std::is_same<typename rawGrad::ElementType, typename rawSOut::ElementType>::value,
                      "Element type mismatch.");
        static_assert(std::is_same<typename rawGrad::DeviceType, typename rawSOut::DeviceType>::value,
                      "Device type mismatch.");

        using ResType = BinaryOp<BinaryOpTags::ColSoftmaxDerivative, rawGrad, rawSOut>;
        return ResType(std::forward<TGrad>(p_grad), std::forward<TSOut>(p_sout));
    }
};

template <typename TGrad, typename TSOut,
          std::enable_if_t<OperColSoftmaxDerivative_<TGrad, TSOut>::valid>* = nullptr>
auto ColSoftmaxDerivative(TGrad&& p_grad, TSOut&& p_sout)
{
    return OperColSoftmaxDerivative_<TGrad, TSOut>::Eval(std::forward<TGrad>(p_grad),
                                                         std::forward<TSOut>(p_sout));
}
}
