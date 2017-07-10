#pragma once

#include <MetaNN/data/dynamic.h>
#include <MetaNN/model_rel/grad_col/grad_collector.h>
#include <stack>
#include <stdexcept>

namespace MetaNN
{
namespace LayerTraits
{
template <typename ElementType, typename DeviceType, typename CateType>
struct LayerInternalBufType_
{
    using tmp2 = DynamicData<ElementType, DeviceType, CateType>;
    using type = std::stack<tmp2, std::list<tmp2>>;
};

template <typename ElementType, typename DeviceType, typename CateType>
using LayerInternalBufType = typename LayerInternalBufType_<ElementType, DeviceType, CateType>::type;    

template <typename TWeight, typename TGrad, typename TGradCollector>
void MatrixGradCollect(const TWeight& weight,
                       TGrad& grad,
                       TGradCollector& col)
{
    while (!grad.empty())
    {
        auto g = grad.top();
        grad.pop();
        col.Collect(weight, g);
    }
}

template <typename TCur>
void NeutralInvariant(const TCur&)
{ }

template <typename TCur, typename TCont>
void NeutralInvariant(const std::stack<TCur, TCont>& cur)
{
    if (!cur.empty())
    {
        throw std::runtime_error("NeutralInvariant Fail!");
    }
}

template <typename TCont> struct Reverse_;

template <template <typename...> class TCont, typename...TElems>
struct Reverse_<TCont<TElems...>>
{
    template <typename TNewCont, typename...T>
    struct imp
    {
        using type = TNewCont;
    };

    template <typename...T1, typename TCur, typename...T2>
    struct imp<TCont<T1...>, TCur, T2...>
    {
        using type = typename imp<TCont<TCur, T1...>, T2...>::type;
    };
    using type = typename imp<TCont<>, TElems...>::type;
};

template <typename TCont>
using Reverse = typename Reverse_<TCont>::type;
}
}