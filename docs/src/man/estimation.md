# Estimation

```MessyTimeSeriesOptim``` provides a generic ECM algorithm easily adaptable to any model that can be written in a linear state-space form. This algorithm is also compatible with penalised maximum likelihood and penalised quasi maximum likelihood estimation.

```@index
Pages = ["estimation.md"]
Depth = 2
```

## ECM algorithm

```@docs
ecm
```

## EstimSettings

```MessyTimeSeriesOptim``` provides the following EstimSettings subtypes.

### Dynamic factor models
```@docs
DFMSettings
```

### Vector autoregressions
```@docs
VARSettings
```
### Vector moving averages
```@docs
VMASettings
```