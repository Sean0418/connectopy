%Matlab code for loading data of structural connectome
clear all; %clear all existing variables in the current environment 
close all; %close all figures

addpath(genpath("./"))

%%%%%%%%%%% load SC data
load HCP_cortical_DesikanAtlas_SC.mat

% This data contains two variables all_id (subject id) and hcp_sc_count
% (68*68*1065, 1065 subjects' structural connectome data) 
% the 68 regions name can be found in BrainRegionName.xlsx


%%%%%%%%%%% load FC data
load HCP_cortical_DesikanAtlas_FC.mat

% This data contains two variables subj_list (subject id) and
% hcp_cortical_fc
% (68*68*1065, 1065 subjects' functional connectome data) 
% the nodes here are same as the ones in SC. 



%load the structrual connectome data into Matlab
load TNPCA_Coeff_HCP_Structural_Connectome.mat

% there are two variables in "PCA_Coeff_HCP_Structural_Connectome.mat"

% network_subject_ids: all subjects id from the HCP dataset that have
% structural network data extracted. Here we have 1065 of them. 

%PCA_Coeff: a matrix of 1*1065*60. We perform TNPCA analysis and reduce each network's size 
%from 68*68 to 60. So you have 1065 vectors with a length of 60, 
%representing the first 60 TNPC coefficients of all subjects in the HCP data. 


%Similarly, you can load the functional connectome into the Matlab
clear all; %clear all existing variables in the current environment 
close all; %close all figures

%load the structrual connectome data into Matlab
load PCA_Coeff_HCP_Functional_Connectome.mat