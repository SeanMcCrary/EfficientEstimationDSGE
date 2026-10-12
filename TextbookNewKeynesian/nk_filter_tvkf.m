function out = nk_filter_tvkf(Y,params,method,options)
% Eva F. Janssens and Sean McCrary
% "Efficient Estimation of Nonlinear DSGE Models"

%NK_FILTER_TVKF Standard or time-varying Kalman filter for current policies.
% METHOD is loglinear, tp_no_zlb, dp_no_zlb, or dp_zlb. Without an observed
% indicator, DP-ZLB selects the branch using the unconstrained policy at the
% forecast mean. With options.binding_indicator, that observed indicator
% selects the conditional measurement branch and its predictive Bernoulli
% probability is computed exclusively from the unconstrained DP rate.

if nargin<4
    options = struct();
end
if ~isnumeric(Y) || ~ismatrix(Y) || isempty(Y) || size(Y,2)~=3 || ~isreal(Y) || ...
        any(isinf(Y(:)))
    error('SimpleNK:FilterData','Y must be a real T-by-3 array.');
end
if ~isfield(options,'dp_horizon'); options.dp_horizon = 40; end
if ~isfield(options,'first_observable'); options.first_observable = 'employment'; end
if ~isfield(options,'include_binding_indicator_likelihood')
    options.include_binding_indicator_likelihood = false;
end
method = lower(char(method));
valid_methods = {'loglinear','tp_no_zlb','dp_no_zlb','dp_zlb'};
if ~any(strcmp(method,valid_methods))
    error('SimpleNK:FilterMethod','Unknown Kalman-filter method %s.',method);
end
if ~isscalar(options.dp_horizon) || options.dp_horizon<1 || ...
        options.dp_horizon~=fix(options.dp_horizon)
    error('SimpleNK:DPHorizon','The DP horizon must be a positive integer.');
end

local_validate_params(params);
T = size(Y,1);
has_observed_indicator = strcmp(method,'dp_zlb') && ...
    isfield(options,'binding_indicator') && ...
    ~isempty(options.binding_indicator);
observed_binding_indicator = [];
if has_observed_indicator
    raw_indicator = options.binding_indicator(:);
    if numel(raw_indicator)~=T || ...
            ~(islogical(raw_indicator) || isnumeric(raw_indicator)) || ...
            ~isreal(raw_indicator) || any(~isfinite(raw_indicator)) || ...
            any(raw_indicator~=0 & raw_indicator~=1)
        error('SimpleNK:BindingIndicator', ...
            'binding_indicator must be a logical or zero-one T-vector.');
    end
    observed_binding_indicator = logical(raw_indicator);
end
if ~isscalar(options.include_binding_indicator_likelihood) || ...
        ~(islogical(options.include_binding_indicator_likelihood) || ...
        ismember(options.include_binding_indicator_likelihood,[0 1]))
    error('SimpleNK:BindingIndicatorLikelihood', ...
        'include_binding_indicator_likelihood must be logical.');
end
include_indicator_likelihood = has_observed_indicator && ...
    logical(options.include_binding_indicator_likelihood);

A = diag([params.rho_z params.rho_d params.rho_m]);
Q = diag([params.sig_z^2 params.sig_d^2 params.sig_m^2]);
H = diag([params.sigMEn^2 params.sigMEpi^2 params.sigMEr^2]);
P0 = diag(diag(Q)./(1-[params.rho_z params.rho_d params.rho_m]'.^2));

[loglinear,params] = nk_loglinear_coefficients(params);
tp_anchor = [];
if strcmp(method,'tp_no_zlb')
    tp_anchor = nk_log_tp_coef(0,0,0,loglinear.vector,params);
end

a_forecast = zeros(T,3);
a_filtered = zeros(T,3);
P_forecast = zeros(3,3,T);
P_filtered = zeros(3,3,T);
y_forecast = zeros(T,3);
y_filtered = zeros(T,3);
loglik_t = zeros(T,1);
bind = false(T,1);
shadow = nan(T,1);
forecast_bind = false(T,1);
predicted_binding_probability = nan(T,1);
regime_loglik_t = zeros(T,1);

at = zeros(3,1);
Pt = P0;
for t = 1:T
    a_forecast(t,:) = at';
    P_forecast(:,:,t) = Pt;
    branch_indicator = [];
    if has_observed_indicator
        branch_indicator = observed_binding_indicator(t);
    end
    [c,Z,bind(t),shadow(t),shadow_slope,forecast_bind(t)] = ...
        local_measurement_map(at,params,method,options, ...
        loglinear,tp_anchor,branch_indicator);
    y_forecast(t,:) = (c+Z*at)';

    if strcmp(method,'dp_zlb')
        shadow_variance = max(shadow_slope*Pt*shadow_slope',0);
        shadow_sd = sqrt(max(shadow_variance,1e-14));
        probability = 0.5*erfc( ...
            (shadow(t)+params.rZLB)/(sqrt(2)*shadow_sd));
        probability = min(max(probability,1e-12),1-1e-12);
        predicted_binding_probability(t) = probability;
        if include_indicator_likelihood
            if observed_binding_indicator(t)
                regime_loglik_t(t) = log(probability);
            else
                regime_loglik_t(t) = log1p(-probability);
            end
        end
    end

    yt = Y(t,:)';
    observed = isfinite(yt);
    if any(observed)
        Zobs = Z(observed,:);
        Hobs = H(observed,observed);
        innovation = yt(observed)-c(observed)-Zobs*at;
        F = Zobs*Pt*Zobs'+Hobs;
        [L,jitter] = local_cholesky(F); %#ok<ASGLU>
        solved = L'\(L\innovation);
        K = (Pt*Zobs')/(F+jitter*eye(size(F)));
        att = at+K*innovation;
        Ptt = Pt-K*Zobs*Pt;
        Ptt = (Ptt+Ptt')/2;
        logdet = 2*sum(log(diag(L)));
        loglik_t(t) = -0.5*(sum(observed)*log(2*pi)+logdet+innovation'*solved);
    else
        att = at;
        Ptt = Pt;
    end
    loglik_t(t) = loglik_t(t)+regime_loglik_t(t);

    a_filtered(t,:) = att';
    P_filtered(:,:,t) = Ptt;
    y_filtered(t,:) = (c+Z*att)';
    at = A*att;
    Pt = A*Ptt*A'+Q;
    Pt = (Pt+Pt')/2;
end

out.method = method;
out.loglik = sum(loglik_t);
out.loglik_t = loglik_t;
out.forecast_state = a_forecast;
out.filtered_state = a_filtered;
out.forecast_covariance = P_forecast;
out.filtered_covariance = P_filtered;
out.forecast_observables = y_forecast;
out.filtered_observables = y_filtered;
% Keep bind as the U-policy forecast selector. branch_bind records the
% observed conditional branch when an indicator is supplied.
out.bind = forecast_bind;
out.branch_bind = bind;
out.forecast_bind = forecast_bind;
out.unconstrained_notional_rate = shadow;
out.predicted_binding_probability = predicted_binding_probability;
out.observed_binding_indicator = observed_binding_indicator;
out.binding_indicator_used = has_observed_indicator;
if strcmp(method,'dp_zlb') && has_observed_indicator
    out.regime_conditioning = 'observed_binding_indicator';
elseif strcmp(method,'dp_zlb')
    out.regime_conditioning = 'unconstrained_forecast_mean';
else
    out.regime_conditioning = 'not_applicable';
end
out.regime_loglik_t = regime_loglik_t;
out.regime_loglik = sum(regime_loglik_t);
out.continuous_likelihood_sample_size = sum(any(isfinite(Y),2));
out.likelihood_sample_size = out.continuous_likelihood_sample_size;
if include_indicator_likelihood
    out.likelihood_sample_size = T;
end
out.first_observable = lower(char(options.first_observable));
end

function [c,Z,bind,shadow,shadow_slope,forecast_bind] = local_measurement_map( ...
        state,params,method,options,loglinear,tp_anchor,branch_indicator)
state = state(:);
bind = false;
shadow = NaN;
shadow_slope = zeros(1,3);
forecast_bind = false;
switch method
    case 'loglinear'
        api = reshape(loglinear.pi,1,3);
        bni = reshape(loglinear.n,1,3);
        api0 = 0; bni0 = 0;
    case 'tp_no_zlb'
        coef = nk_log_tp_coef(state(1),state(2),state(3),tp_anchor,params);
        coef = coef(:);
        api0 = coef(1); api = reshape(coef(2:4),1,3);
        bni0 = coef(5); bni = reshape(coef(6:8),1,3);
    case 'dp_no_zlb'
        coef = nk_nozlb_dp_coef(state,params,options.dp_horizon);
        coef = coef(:);
        api = reshape(coef(2:4),1,3);
        bni = reshape(coef(6:8),1,3);
        api0 = coef(1)-api*state;
        bni0 = coef(5)-bni*state;
    case 'dp_zlb'
        solution = nk_zlb_dp_coef(state,params,options.dp_horizon);
        allcoef = solution.coef(:);
        piU = allcoef(1); nU = allcoef(5);
        shadow = params.phi_pi*piU+params.phi_y*(nU+state(1))+state(3);
        apiU = reshape(allcoef(2:4),1,3);
        bniU = reshape(allcoef(6:8),1,3);
        shadow_slope = params.phi_pi*apiU ...
            +params.phi_y*(bniU+[1 0 0])+[0 0 1];
        forecast_bind = shadow<=-params.rZLB;
        if forecast_bind~=logical(solution.bind)
            error('SimpleNK:DPSelector', ...
                'DP binding must be determined by the unconstrained policy only.');
        end
        bind = forecast_bind;
        if ~isempty(branch_indicator)
            bind = logical(branch_indicator);
        end
        if bind
            coef = allcoef(9:16);
        else
            coef = allcoef(1:8);
        end
        api = reshape(coef(2:4),1,3);
        bni = reshape(coef(6:8),1,3);
        api0 = coef(1)-api*state;
        bni0 = coef(5)-bni*state;
end

if ~isequal(size(api),[1 3]) || ~isequal(size(bni),[1 3])
    error('SimpleNK:MeasurementSlopeShape', ...
        'Policy slopes must be row vectors with three state derivatives.');
end

if strcmpi(options.first_observable,'employment')
    first0 = bni0;
    first_slope = bni;
elseif strcmpi(options.first_observable,'output')
    first0 = bni0;
    first_slope = bni+[1 0 0];
else
    error('SimpleNK:ObservationMap', ...
        'first_observable must be employment or output.');
end

c = zeros(3,1);
Z = zeros(3,3);
c(1) = first0; Z(1,:) = first_slope;
c(2) = api0;   Z(2,:) = api;
if strcmp(method,'dp_zlb') && bind
    c(3) = -params.rZLB;
else
    c(3) = params.phi_pi*api0+params.phi_y*bni0;
    Z(3,:) = params.phi_pi*api+params.phi_y*(bni+[1 0 0])+[0 0 1];
end
end

function [L,jitter] = local_cholesky(F)
scale = max(1,max(abs(diag(F))));
for exponent = [-14 -12 -10 -8 -6]
    jitter = scale*10^exponent;
    [L,p] = chol((F+F')/2+jitter*eye(size(F)),'lower');
    if p==0
        return
    end
end
error('SimpleNK:FilterCovariance','Innovation covariance is not positive definite.');
end

function local_validate_params(params)
required = {'tau','beta','eta','kappa','phi_pi','phi_y', ...
    'rho_z','rho_d','rho_m','sig_z','sig_d','sig_m','rZLB', ...
    'sigMEn','sigMEpi','sigMEr'};
if ~isstruct(params) || ~isscalar(params) || ~all(isfield(params,required))
    error('SimpleNK:FilterParameters', ...
        'The filter parameter structure must be specified completely.');
end
for j = 1:numel(required)
    value = params.(required{j});
    if ~isscalar(value) || ~isreal(value) || ~isfinite(value)
        error('SimpleNK:FilterParameters', ...
            'Parameter %s must be a finite real scalar.',required{j});
    end
end
if params.tau<=0 || params.beta<=0 || params.beta>=1 || ...
        params.eta<=0 || params.kappa<=0 || params.phi_pi<=0 || ...
        params.phi_y<0 || params.rZLB<=0 || ...
        any(abs([params.rho_z params.rho_d params.rho_m])>=1) || ...
        any([params.sig_z params.sig_d params.sig_m ...
        params.sigMEn params.sigMEpi params.sigMEr]<=0)
    error('SimpleNK:FilterParameters', ...
        'A filter parameter is outside its admissible range.');
end
end
