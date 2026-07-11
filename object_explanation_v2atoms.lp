%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% object_explanation_v2atoms.lp
% ASP explanation selection rules for decomposev2.py visual atoms.
%
% Expected V2 facts:
%   detection(O).
%   class(O,C).
%   candidate(P,O).
%   bbox(P,X,Y,W,H).
%
% Atom types:
%   semantic_region(P).
%   support_perimeter(P).
%   geometric_primitive(P).
%   boundary_fragment(P).
%   thin_extension(P).
%
% Evidence:
%   support_mean(P,S). support_max(P,S). compactness(P,S).
%   rectangularity(P,S). perimeter_smoothness(P,S).
%   edge_confidence(P,S). side_contrast(P,S).
%   near_support_perimeter(P,S). fit_error(P,E).
%
% Learned facts may be supplied separately as:
%   bound(Class, semantic_region, L, U).
%   weight(Class, Feature, W).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% ---------------------------------------------------------------------------
% Candidate primitives
% ---------------------------------------------------------------------------

candidate_primitive(P,O) :-
    candidate(P,O).

candidate_primitive(P,O) :-
    semantic_region(P),
    candidate(P,O).

candidate_primitive(P,O) :-
    support_perimeter(P),
    candidate(P,O).

candidate_primitive(P,O) :-
    geometric_primitive(P),
    candidate(P,O).

candidate_primitive(P,O) :-
    boundary_fragment(P),
    candidate(P,O).

candidate_primitive(P,O) :-
    thin_extension(P),
    candidate(P,O).


% ---------------------------------------------------------------------------
% Selection
% ---------------------------------------------------------------------------

{ selected(P,O) } :-
    candidate_primitive(P,O).


% Keep old name for compatibility with your previous pipeline.
explains(P,O) :-
    selected(P,O).


% ---------------------------------------------------------------------------
% Effective bounds
% ---------------------------------------------------------------------------

default_bound(semantic_region,1,4).

has_bound(C,T) :-
    bound(C,T,_,_).

eff_bound(C,T,L,U) :-
    bound(C,T,L,U).

eff_bound(C,T,L,U) :-
    class(_,C),
    default_bound(T,L,U),
    not has_bound(C,T).


% Semantic region cardinality.
L { selected(P,O) :
        semantic_region(P),
        candidate(P,O)
  } U :-
    detection(O),
    class(O,C),
    eff_bound(C,semantic_region,L,U).


% ---------------------------------------------------------------------------
% Structural requirements
% ---------------------------------------------------------------------------

included_body(O) :-
    selected(P,O),
    semantic_region(P),
    region_role(P,body).

% Every explanation must include at least one body/support region.
:- detection(O), not included_body(O).


% If an atom is derived from a region, require that source region to be selected.
% This keeps perimeter/geometric atoms grounded in a selected body region.
:- selected(P,O),
   from_region(P,R),
   candidate(R,O),
   not selected(R,O).


% Thin extensions should be attached to a selected region.
:- selected(T,O),
   thin_extension(T),
   attached_to(T,R),
   candidate(R,O),
   not selected(R,O).


% ---------------------------------------------------------------------------
% Positive features
% ---------------------------------------------------------------------------

feature(P,semantic_region,100) :-
    semantic_region(P).

feature(P,support_perimeter,100) :-
    support_perimeter(P).

feature(P,geometric_primitive,100) :-
    geometric_primitive(P).

feature(P,boundary_fragment,100) :-
    boundary_fragment(P).

feature(P,thin_extension,100) :-
    thin_extension(P).


feature(P,support_mean,S) :-
    support_mean(P,S).

feature(P,support_max,S) :-
    support_max(P,S).

feature(P,compactness,S) :-
    compactness(P,S).

feature(P,rectangularity,S) :-
    rectangularity(P,S).

feature(P,perimeter_smoothness,S) :-
    perimeter_smoothness(P,S).

feature(P,edge_confidence,S) :-
    edge_confidence(P,S).

feature(P,side_contrast,S) :-
    side_contrast(P,S).

feature(P,near_support_perimeter,S) :-
    near_support_perimeter(P,S).


feature(P,region_role(Role),100) :-
    semantic_region(P),
    region_role(P,Role).

feature(P,primitive_type(Type),100) :-
    geometric_primitive(P),
    primitive_type(P,Type).

feature(P,boundary_role(Role),100) :-
    boundary_fragment(P),
    boundary_role(P,Role).


% Useful named features for Optuna to learn.
feature(P,body_region,100) :-
    semantic_region(P),
    region_role(P,body).

feature(P,outer_boundary,100) :-
    boundary_fragment(P),
    boundary_role(P,outer_candidate).

feature(P,geometry_from_body,100) :-
    geometric_primitive(P),
    from_region(P,R),
    region_role(R,body).

feature(P,perimeter_from_body,100) :-
    support_perimeter(P),
    from_region(P,R),
    region_role(R,body).

feature(P,attached_thin_extension,100) :-
    thin_extension(P),
    attached_to(P,R),
    region_role(R,body).


% ---------------------------------------------------------------------------
% Penalty features
% ---------------------------------------------------------------------------

penalty_feature(P,fit_error,E) :-
    fit_error(P,E).

penalty_feature(P,complexity,1) :-
    candidate_primitive(P,_).

penalty_feature(P,weak_boundary,V) :-
    boundary_fragment(P),
    edge_confidence(P,E),
    V = 100 - E.

penalty_feature(P,low_side_contrast,V) :-
    boundary_fragment(P),
    side_contrast(P,S),
    V = 100 - S.

penalty_feature(P,far_from_support_perimeter,V) :-
    boundary_fragment(P),
    near_support_perimeter(P,S),
    V = 100 - S.

penalty_feature(P,large_area,V) :-
    semantic_region(P),
    area_ratio(P,A),
    A > 85,
    V = A - 85.

penalty_feature(P,small_area,V) :-
    semantic_region(P),
    area_ratio(P,A),
    A < 10,
    V = 10 - A.


% Optional pairwise redundancy if overlap(P1,P2,V) facts exist.
pair_penalty(P1,P2,redundancy,V) :-
    selected(P1,O),
    selected(P2,O),
    overlap(P1,P2,V),
    P1 < P2.


% ---------------------------------------------------------------------------
% Effective weights
% ---------------------------------------------------------------------------

has_weight(C,F) :-
    weight(C,F,_).

eff_weight(C,F,W) :-
    weight(C,F,W).

eff_weight(C,F,W) :-
    class(_,C),
    default_weight(F,W),
    not has_weight(C,F).


% ---------------------------------------------------------------------------
% Initial default weights
% These are seed/default weights. Optuna should later write learned weight(C,F,W)
% facts per class, which override these defaults.
% ---------------------------------------------------------------------------

default_weight(semantic_region,1).
default_weight(support_perimeter,2).
default_weight(geometric_primitive,1).
default_weight(boundary_fragment,1).
default_weight(thin_extension,3).

default_weight(support_mean,2).
default_weight(support_max,1).
default_weight(compactness,1).
default_weight(rectangularity,1).
default_weight(perimeter_smoothness,1).

default_weight(edge_confidence,2).
default_weight(side_contrast,4).
default_weight(near_support_perimeter,3).

default_weight(region_role(body),5).
default_weight(body_region,5).

default_weight(primitive_type(ellipse),1).
default_weight(primitive_type(min_rect),1).
default_weight(primitive_type(convex_hull),1).

default_weight(boundary_role(outer_candidate),4).
default_weight(outer_boundary,5).

default_weight(geometry_from_body,2).
default_weight(perimeter_from_body,3).
default_weight(attached_thin_extension,4).

default_weight(fit_error,2).
default_weight(complexity,2).
default_weight(weak_boundary,2).
default_weight(low_side_contrast,3).
default_weight(far_from_support_perimeter,2).
default_weight(large_area,1).
default_weight(small_area,1).
default_weight(redundancy,2).


% ---------------------------------------------------------------------------
% Optimization
% ---------------------------------------------------------------------------

#maximize {
    W*S@3,O,P,F :
        selected(P,O),
        feature(P,F,S),
        class(O,C),
        eff_weight(C,F,W)
}.

#minimize {
    W*E@3,O,P,F :
        selected(P,O),
        penalty_feature(P,F,E),
        class(O,C),
        eff_weight(C,F,W)
}.

#minimize {
    W*V@2,O,P1,P2,F :
        pair_penalty(P1,P2,F,V),
        class(O,C),
        eff_weight(C,F,W)
}.


% ---------------------------------------------------------------------------
% Output
% ---------------------------------------------------------------------------

#show explains/2.
#show selected/2.
#show included_body/1.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extra V2 compactness constraints
% Added after initial smoke test: prevent selecting every atom.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Type-specific default bounds for compact explanations.
default_bound(support_perimeter,0,2).
default_bound(geometric_primitive,0,3).
default_bound(boundary_fragment,0,4).
default_bound(thin_extension,0,3).

% Support perimeter cardinality.
L { selected(P,O) :
        support_perimeter(P),
        candidate(P,O)
  } U :-
    detection(O),
    class(O,C),
    eff_bound(C,support_perimeter,L,U).

% Geometry cardinality.
L { selected(P,O) :
        geometric_primitive(P),
        candidate(P,O)
  } U :-
    detection(O),
    class(O,C),
    eff_bound(C,geometric_primitive,L,U).

% Boundary cardinality.
L { selected(P,O) :
        boundary_fragment(P),
        candidate(P,O)
  } U :-
    detection(O),
    class(O,C),
    eff_bound(C,boundary_fragment,L,U).

% Thin-extension cardinality.
L { selected(P,O) :
        thin_extension(P),
        candidate(P,O)
  } U :-
    detection(O),
    class(O,C),
    eff_bound(C,thin_extension,L,U).

% Hard cap on total explanation size.
:- detection(O), #count { P : selected(P,O) } > 10.


% Optional learned total explanation cap.
:- detection(O),
   class(O,C),
   max_selected(C,M),
   #count { P : selected(P,O) } > M.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V2 output-box candidate selection
% The final refined box is chosen_box(B,O), not union(all selected atoms).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Choose exactly one output box candidate.
1 { chosen_box(B,O) : box_candidate(B,O) } 1 :-
    detection(O).

% If a chosen box is built from atoms, force those atoms into the explanation.
selected(P,O) :-
    chosen_box(B,O),
    cand_source(B,P).

% ---------------- Output-box features ----------------

box_feature(B,box_type(T),100) :-
    cand_type(B,T).

box_feature(B,source_region,100) :-
    cand_source(B,P),
    semantic_region(P).

box_feature(B,source_perimeter,100) :-
    cand_source(B,P),
    support_perimeter(P).

box_feature(B,source_geometry,100) :-
    cand_source(B,P),
    geometric_primitive(P).

box_feature(B,source_boundary,100) :-
    cand_source(B,P),
    boundary_fragment(P).

box_feature(B,source_outer_boundary,100) :-
    cand_source(B,P),
    boundary_fragment(P),
    boundary_role(P,outer_candidate).

box_feature(B,source_thin_extension,100) :-
    cand_source(B,P),
    thin_extension(P).

box_feature(B,source_support_mean,S) :-
    cand_source(B,P),
    support_mean(P,S).

box_feature(B,source_edge_confidence,S) :-
    cand_source(B,P),
    edge_confidence(P,S).

box_feature(B,source_side_contrast,S) :-
    cand_source(B,P),
    side_contrast(P,S).

box_feature(B,source_near_support_perimeter,S) :-
    cand_source(B,P),
    near_support_perimeter(P,S).

% ---------------- Output-box penalties ----------------

box_penalty(B,candidate_complexity,N) :-
    cand_n_sources(B,N).

box_penalty(B,candidate_large_area,V) :-
    cand_area_ratio(B,A),
    A > 180,
    V = A - 180.

box_penalty(B,candidate_small_area,V) :-
    cand_area_ratio(B,A),
    A < 40,
    V = 40 - A.

% ---------------- Default output-box weights ----------------

default_weight(box_type(proposal),4).
default_weight(box_type(region),2).
default_weight(box_type(perimeter),3).
default_weight(box_type(thin),3).
default_weight(box_type(boundary),1).
default_weight(box_type(outer_boundary),4).

default_weight(box_type(geom_ellipse),2).
default_weight(box_type(geom_min_rect),2).
default_weight(box_type(geom_convex_hull),1).

default_weight(box_type(union_proposal_thin),5).
default_weight(box_type(union_proposal_outer),4).
default_weight(box_type(union_proposal_thin_outer),4).
default_weight(box_type(union_region_thin),5).
default_weight(box_type(union_region_outer),4).
default_weight(box_type(union_region_thin_outer),4).

default_weight(source_region,2).
default_weight(source_perimeter,2).
default_weight(source_geometry,1).
default_weight(source_boundary,1).
default_weight(source_outer_boundary,4).
default_weight(source_thin_extension,4).

default_weight(source_support_mean,1).
default_weight(source_edge_confidence,1).
default_weight(source_side_contrast,3).
default_weight(source_near_support_perimeter,3).

default_weight(candidate_complexity,2).
default_weight(candidate_large_area,3).
default_weight(candidate_small_area,2).

% ---------------- Output-box optimization ----------------
% Higher priority than atom-selection optimization.
#maximize {
    W*S@5,O,B,F :
        chosen_box(B,O),
        box_feature(B,F,S),
        class(O,C),
        eff_weight(C,F,W)
}.

#minimize {
    W*E@5,O,B,F :
        chosen_box(B,O),
        box_penalty(B,F,E),
        class(O,C),
        eff_weight(C,F,W)
}.

#show chosen_box/2.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V2 chosen-box reliability features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hard safety constraints for non-proposal output boxes.
:- chosen_box(B,O),
   cand_type(B,T),
   T != proposal,
   cand_iou_proposal(B,S),
   S < 15.

:- chosen_box(B,O),
   cand_type(B,T),
   T != proposal,
   cand_area_ratio(B,A),
   A < 25.

:- chosen_box(B,O),
   cand_type(B,T),
   T != proposal,
   cand_area_ratio(B,A),
   A > 300.

:- chosen_box(B,O),
   cand_type(B,T),
   T != proposal,
   cand_center_shift(B,S),
   S > 90.

% Positive reliability features.
box_feature(B,candidate_is_proposal,100) :-
    cand_type(B,proposal).

box_feature(B,candidate_iou_proposal,S) :-
    cand_iou_proposal(B,S).

box_feature(B,candidate_area_near_proposal,S) :-
    cand_area_delta(B,D),
    S = 100 - D,
    D <= 100.

box_feature(B,candidate_low_source_count,S) :-
    cand_n_sources(B,N),
    S = 100 - 10*N,
    N <= 10.

% Penalty reliability features.
box_penalty(B,candidate_low_iou_proposal,V) :-
    cand_iou_proposal(B,S),
    V = 100 - S.

box_penalty(B,candidate_center_shift,S) :-
    cand_center_shift(B,S).

box_penalty(B,candidate_area_delta,D) :-
    cand_area_delta(B,D).

box_penalty(B,candidate_many_sources,N) :-
    cand_n_sources(B,N),
    N > 3.

% Default reliability weights.
default_weight(candidate_is_proposal,8).
default_weight(candidate_iou_proposal,4).
default_weight(candidate_area_near_proposal,3).
default_weight(candidate_low_source_count,2).

default_weight(candidate_low_iou_proposal,5).
default_weight(candidate_center_shift,6).
default_weight(candidate_area_delta,3).
default_weight(candidate_many_sources,3).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% V2 chosen-box reliability features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Hard safety constraints for non-proposal output boxes.
:- chosen_box(B,O),
   cand_type(B,T),
   T != proposal,
   cand_iou_proposal(B,S),
   S < 15.

:- chosen_box(B,O),
   cand_type(B,T),
   T != proposal,
   cand_area_ratio(B,A),
   A < 25.

:- chosen_box(B,O),
   cand_type(B,T),
   T != proposal,
   cand_area_ratio(B,A),
   A > 300.

:- chosen_box(B,O),
   cand_type(B,T),
   T != proposal,
   cand_center_shift(B,S),
   S > 90.

% Positive reliability features.
box_feature(B,candidate_is_proposal,100) :-
    cand_type(B,proposal).

box_feature(B,candidate_iou_proposal,S) :-
    cand_iou_proposal(B,S).

box_feature(B,candidate_area_near_proposal,S) :-
    cand_area_delta(B,D),
    S = 100 - D,
    D <= 100.

box_feature(B,candidate_low_source_count,S) :-
    cand_n_sources(B,N),
    S = 100 - 10*N,
    N <= 10.

% Penalty reliability features.
box_penalty(B,candidate_low_iou_proposal,V) :-
    cand_iou_proposal(B,S),
    V = 100 - S.

box_penalty(B,candidate_center_shift,S) :-
    cand_center_shift(B,S).

box_penalty(B,candidate_area_delta,D) :-
    cand_area_delta(B,D).

box_penalty(B,candidate_many_sources,N) :-
    cand_n_sources(B,N),
    N > 3.

% Default reliability weights.
default_weight(candidate_is_proposal,8).
default_weight(candidate_iou_proposal,4).
default_weight(candidate_area_near_proposal,3).
default_weight(candidate_low_source_count,2).

default_weight(candidate_low_iou_proposal,5).
default_weight(candidate_center_shift,6).
default_weight(candidate_area_delta,3).
default_weight(candidate_many_sources,3).

