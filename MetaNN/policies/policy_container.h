#pragma once

namespace MetaNN
{
template <typename...TPolicies>
struct PolicyContainer;

template <typename T>
constexpr static bool IsPolicyContainer = false;

template <typename...T>
constexpr static bool IsPolicyContainer<PolicyContainer<T...>> = true;

template <typename TLayerName, typename...TPolicies>
struct SubPolicyContainer;

template <typename T>
constexpr static bool IsSubPolicyContainer = false;

template <typename TLayer, typename...T>
constexpr static bool IsSubPolicyContainer<SubPolicyContainer<TLayer, T...>> = true;
}