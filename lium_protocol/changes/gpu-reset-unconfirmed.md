minor
`GpuResetErrorCode.ResetUnconfirmed`: a renter's GPU reset ran, but the clock reset after it gave no answer, so it could not be confirmed. Today the validator reports that case as `ClockLockNotSupported`, which means the machine cannot lock clocks at all. The validator sends the new code only once the backend accepts it; until then it keeps sending `ClockLockNotSupported`.
