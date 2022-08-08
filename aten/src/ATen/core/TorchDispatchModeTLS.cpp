#include <ATen/core/TorchDispatchModeTLS.h>
#include <c10/core/SafePyObject.h>
#include <c10/core/DispatchKeySet.h>

namespace at { namespace impl {

thread_local std::shared_ptr<SafePyObject> torchDispatchModeState;
thread_local bool torchDispatchModeSkipNext;

void TorchDispatchModeTLS::set_state(std::shared_ptr<SafePyObject> state) {
  if (state) {
    c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, true);
    c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonTLSSnapshot, true);
  } else {
    TorchDispatchModeTLS::reset_state();
  }
  torchDispatchModeState = std::move(state);
}

const std::shared_ptr<SafePyObject>& TorchDispatchModeTLS::get_state() {
  return torchDispatchModeState;
}

void TorchDispatchModeTLS::reset_state() {
  torchDispatchModeState.reset();
  c10::impl::tls_set_dispatch_key_included(DispatchKey::Python, false);
  c10::impl::tls_set_dispatch_key_included(DispatchKey::PythonTLSSnapshot, false);
}

bool TorchDispatchModeTLS::exchange_skip_next(bool new_skip_next) {
  return std::exchange(torchDispatchModeSkipNext, new_skip_next);
}

bool TorchDispatchModeTLS::peek_skip_next() {
  return torchDispatchModeSkipNext;
}

bool dispatch_mode_enabled() {
  return static_cast<bool>(at::impl::TorchDispatchModeTLS::get_state());
}

bool tensor_has_dispatch(const at::Tensor& t) {
  DispatchKeySet key_set({DispatchKey::Python, DispatchKey::PythonTLSSnapshot});
  return t.key_set().has_any(key_set);
}

bool tensorlist_has_dispatch(const at::TensorList& li) {
  for (const auto& t: li) {
    if (tensor_has_dispatch(t)) {
      return true;
    }
  }
  return false;
}

bool tensorlist_has_dispatch(const c10::List<c10::optional<at::Tensor>>& li) {
  for (auto i : c10::irange(li.size())) {
    auto t = li.get(i);
    if (t && tensor_has_dispatch(*t)) {
      return true;
    }
  }
  return false;
}

} // namespace impl
} // namespace at
