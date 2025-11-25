%% summarize_hcp_connectome.m
% Clean loader + summary for SC, FC, and TN-PCA

clear; clc; close all;

%% ---------- Load structural connectome (SC) ----------
load('HCP_cortical_DesikanAtlas_SC.mat');
% Expected variables:
%   all_id       : [1065 x 1] subject IDs
%   hcp_sc_count : [68 x 68 x 1065] structural connectomes

[n_regions_sc, ~, n_subj_sc] = size(hcp_sc_count);

%% ---------- Load functional connectome (FC) ----------
load('HCP_cortical_DesikanAtlas_FC.mat');
% Expected variables:
%   subj_list       : [1065 x 1] IDs for attempted FC
%   hcp_cortical_fc : cell array; some cells may be empty

if isnumeric(hcp_cortical_fc)
    % Already numeric 68x68xN
    fc_array    = hcp_cortical_fc;
    [n_regions_fc, ~, n_subj_fc] = size(fc_array);
    subj_list_fc = subj_list(:);

elseif iscell(hcp_cortical_fc)
    % Cell array, some entries empty -> keep only non-empty FCs
    all_cells   = hcp_cortical_fc(:);
    is_nonempty = ~cellfun(@isempty, all_cells);
    nonempty_idx = find(is_nonempty);

    if isempty(nonempty_idx)
        error('hcp_cortical_fc: all cells are empty – no FC data.');
    end

    % Use first non-empty cell as template
    first_idx = nonempty_idx(1);
    template  = all_cells{first_idx};
    [n_regions_fc, ~] = size(template);

    n_subj_fc   = numel(nonempty_idx);
    fc_array    = zeros(n_regions_fc, n_regions_fc, n_subj_fc);

    % only IDs with valid FC data
    subj_list_fc = subj_list(nonempty_idx);

    for k = 1:n_subj_fc
        fc_mat = all_cells{nonempty_idx(k)};
        if isempty(fc_mat)
            error('Unexpected empty FC matrix at nonempty_idx(%d).', k);
        end
        [r, c] = size(fc_mat);
        if r ~= n_regions_fc || c ~= n_regions_fc
            error('Inconsistent FC matrix size at index %d: got %dx%d, expected %dx%d.', ...
                nonempty_idx(k), r, c, n_regions_fc, n_regions_fc);
        end
        fc_array(:, :, k) = fc_mat;
    end

else
    error('hcp_cortical_fc is of unsupported type: %s', class(hcp_cortical_fc));
end

%% ---------- Load TN-PCA coefficients ----------
% Structural
tmp_sc = load('TNPCA_Coeff_HCP_Structural_Connectome.mat');
disp('Fields in structural TN-PCA file:');
disp(fieldnames(tmp_sc));

% Try to detect coefficient + ID fields in a generic way
fn_sc = fieldnames(tmp_sc);

% Heuristic: PCA coeffs are the largest numeric 3D array
pca_idx_sc = [];
id_idx_sc  = [];
for i = 1:numel(fn_sc)
    v = tmp_sc.(fn_sc{i});
    if isnumeric(v) && ndims(v) == 3
        pca_idx_sc = i;
    elseif isnumeric(v) && isvector(v)
        id_idx_sc = i;
    end
end

if isempty(pca_idx_sc)
    error('Could not find 3D PCA_Coeff in structural TN-PCA file.');
end

pca_coeff_sc_raw = tmp_sc.(fn_sc{pca_idx_sc}); % e.g., [1 x 1065 x 60] or [1065 x 60]
if isempty(id_idx_sc)
    warning('No explicit subject ID vector found in structural TN-PCA file.');
    network_subject_ids_sc = [];
else
    network_subject_ids_sc = tmp_sc.(fn_sc{id_idx_sc});
end

% Squeeze coefficients to N x K
coeff_sc = squeeze(pca_coeff_sc_raw);  % will be [1065 x 60] or [60 x 1065]
if size(coeff_sc, 1) < size(coeff_sc, 2)
    coeff_sc = coeff_sc.';            % ensure rows = subjects
end
[n_subj_tn_sc, n_comp_sc] = size(coeff_sc);

% Functional TN-PCA
tmp_fc = load('TNPCA_Coeff_HCP_Functional_Connectome.mat');
disp('Fields in functional TN-PCA file:');
disp(fieldnames(tmp_fc));

fn_fc = fieldnames(tmp_fc);
pca_idx_fc = [];
id_idx_fc  = [];
for i = 1:numel(fn_fc)
    v = tmp_fc.(fn_fc{i});
    if isnumeric(v) && ndims(v) == 3
        pca_idx_fc = i;
    elseif isnumeric(v) && isvector(v)
        id_idx_fc = i;
    end
end

if isempty(pca_idx_fc)
    error('Could not find 3D PCA_Coeff in functional TN-PCA file.');
end

pca_coeff_fc_raw = tmp_fc.(fn_fc{pca_idx_fc});
if isempty(id_idx_fc)
    warning('No explicit subject ID vector found in functional TN-PCA file.');
    network_subject_ids_fc = [];
else
    network_subject_ids_fc = tmp_fc.(fn_fc{id_idx_fc});
end

coeff_fc = squeeze(pca_coeff_fc_raw);
if size(coeff_fc, 1) < size(coeff_fc, 2)
    coeff_fc = coeff_fc.';
end
[n_subj_tn_fc, n_comp_fc] = size(coeff_fc);

%% ---------- BASIC SHAPES ----------
fprintf('================= BASIC SHAPES =================\n');
fprintf('Structural connectome (SC): %d regions x %d regions x %d subjects\n', ...
    n_regions_sc, n_regions_sc, n_subj_sc);
fprintf('Functional connectome (FC): %d regions x %d regions x %d subjects (non-empty)\n', ...
    n_regions_fc, n_regions_fc, n_subj_fc);

fprintf('\nTN-PCA (structural): %d subjects x %d components\n', ...
    n_subj_tn_sc, n_comp_sc);
fprintf('TN-PCA (functional): %d subjects x %d components\n', ...
    n_subj_tn_fc, n_comp_fc);

%% ---------- SUBJECT ID CHECKS ----------
fprintf('\n================= SUBJECT ID CHECKS =============\n');

ids_sc = all_id(:);
ids_fc = subj_list_fc(:);  % NOTE: filtered FC subjects only

common_ids_sc_fc = intersect(ids_sc, ids_fc);
fprintf('Subjects in SC      : %d\n', numel(ids_sc));
fprintf('Subjects in FC      : %d\n', numel(ids_fc));
fprintf('Intersection SC & FC: %d subjects\n', numel(common_ids_sc_fc));

if numel(common_ids_sc_fc) == numel(ids_fc)
    fprintf('-> All FC subjects are also in SC.\n');
end

if numel(common_ids_sc_fc) == numel(ids_sc)
    fprintf('-> All SC subjects have non-empty FC.\n');
else
    fprintf('-> Some SC subjects are missing non-empty FC.\n');
end

% TN-PCA ID counts (if present)
if ~isempty(network_subject_ids_sc)
    ids_tn_sc = network_subject_ids_sc(:);
    fprintf('\nTN-PCA (structural) subjects (ID vector length): %d\n', numel(ids_tn_sc));
    fprintf('TN-PCA struct IDs match SC IDs (as set)? %d\n', ...
        isequal(sort(ids_tn_sc), sort(ids_sc)));
else
    fprintf('\nNo structural TN-PCA ID vector found; cannot compare IDs.\n');
end

if ~isempty(network_subject_ids_fc)
    ids_tn_fc = network_subject_ids_fc(:);
    fprintf('TN-PCA (functional) subjects (ID vector length): %d\n', numel(ids_tn_fc));
    fprintf('TN-PCA funct IDs match FC IDs (as set)?  %d\n', ...
        isequal(sort(ids_tn_fc), sort(ids_fc)));
else
    fprintf('No functional TN-PCA ID vector found; cannot compare IDs.\n');
end

%% ---------- STRUCTURAL CONNECTOME STATS ----------
fprintf('\n================= STRUCTURAL CONNECTOME (SC) ====\n');
sc_vals = double(hcp_sc_count(:));
fprintf('SC: total entries           : %d\n', numel(sc_vals));
fprintf('SC: non-zero entries        : %d (%.2f%%)\n', nnz(sc_vals), ...
    100 * nnz(sc_vals) / numel(sc_vals));
fprintf('SC: min = %.2f, max = %.2f\n', min(sc_vals), max(sc_vals));
fprintf('SC: mean = %.2f, std = %.2f\n', mean(sc_vals), std(sc_vals));

idx_triu = find(triu(ones(n_regions_sc), 1));
density_sc = zeros(n_subj_sc, 1);
for s = 1:n_subj_sc
    A = hcp_sc_count(:, :, s);
    edges = A(idx_triu);
    density_sc(s) = mean(edges > 0);
end
fprintf('SC edge density (edges > 0, upper triangle only):\n');
fprintf('  mean = %.3f, std = %.3f, min = %.3f, max = %.3f\n', ...
    mean(density_sc), std(density_sc), min(density_sc), max(density_sc));

%% ---------- FUNCTIONAL CONNECTOME STATS ----------
fprintf('\n================= FUNCTIONAL CONNECTOME (FC) ====\n');
fc_vals = fc_array(:);
fprintf('FC: total entries         : %d\n', numel(fc_vals));
fprintf('FC: min = %.3f, max = %.3f\n', min(fc_vals), max(fc_vals));
fprintf('FC: mean = %.3f, std = %.3f\n', mean(fc_vals), std(fc_vals));

mask_offdiag = ~eye(n_regions_fc);
fc_offdiag_vals = fc_array(repmat(mask_offdiag, [1, 1, n_subj_fc]));
fc_offdiag_vals = fc_offdiag_vals(:);
fprintf('FC (off-diagonal only): min = %.3f, max = %.3f, mean = %.3f, std = %.3f\n', ...
    min(fc_offdiag_vals), max(fc_offdiag_vals), mean(fc_offdiag_vals), std(fc_offdiag_vals));

%% ---------- TN-PCA COEFFICIENT STATS ----------
fprintf('\n================= TN-PCA COEFFICIENTS ==========\n');
mean_sc_comp = mean(coeff_sc, 1);
std_sc_comp  = std(coeff_sc, 0, 1);
fprintf('First 5 TN-PCA components (structural):\n');
for k = 1:min(5, n_comp_sc)
    fprintf('  Comp %2d: mean = %+7.3f, std = %7.3f\n', k, mean_sc_comp(k), std_sc_comp(k));
end

mean_fc_comp = mean(coeff_fc, 1);
std_fc_comp  = std(coeff_fc, 0, 1);
fprintf('\nFirst 5 TN-PCA components (functional):\n');
for k = 1:min(5, n_comp_fc)
    fprintf('  Comp %2d: mean = %+7.3f, std = %7.3f\n', k, mean_fc_comp(k), std_fc_comp(k));
end

fprintf('\n=========== DONE. Summary complete. ==========\n');
