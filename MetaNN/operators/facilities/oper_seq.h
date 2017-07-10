#pragma once

namespace MetaNN
{
template <typename...TCases>
struct OperSeqContainer;

template <typename TOpTag>
struct OperBuildInSeq_;

template <typename TOpTag>
using OperBuildInSeq = typename OperBuildInSeq_<TOpTag>::type;

template <typename TOpTag>
struct OperSeq_
{
    using type = OperBuildInSeq<TOpTag>;
};

template <typename TOpTag>
using OperSeq = typename OperSeq_<TOpTag>::type;

template <typename TOperSeqCont, typename...TCases>
struct PushOperSeq_;

template <typename...TOriCases, typename...TCases>
struct PushOperSeq_<OperSeqContainer<TOriCases...>, TCases...>
{
    using type = OperSeqContainer<TCases..., TOriCases...>;
};

template <typename TOperSeqCont, typename...TCases>
using PushOperSeq = typename PushOperSeq_<TOperSeqCont, TCases...>::type;

template <typename TOperSeqCont>
struct OperSeqHead_;

template <typename TH, typename...TCases>
struct OperSeqHead_<OperSeqContainer<TH, TCases...>>
{
    using type = TH;
};

template <typename TOperSeqCont>
using OperSeqHead = typename OperSeqHead_<TOperSeqCont>::type;

template <typename TOperSeqCont>
struct OperSeqTail_;

template <typename TH, typename...TCases>
struct OperSeqTail_<OperSeqContainer<TH, TCases...>>
{
    using type = OperSeqContainer<TCases...>;
};

template <typename TOperSeqCont>
using OperSeqTail = typename OperSeqTail_<TOperSeqCont>::type;
}