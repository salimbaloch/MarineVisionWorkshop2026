
detection(o1).
class(o1,animal_urchin).
proposal_box(o1,box(18,17,132,126)).

% Semantic Regions
semantic_region(r1).
belongs_to_detection(r1,o1).
region_role(r1,body).
region_centroid(r1,point(71,66)).
region_box(r1,box(16,6,127,127)).
region_area(r1,10490).
semantic_support(r1,78).
region_objectness(r1,85).
region_coherence(r1,93).
confidence(r1,84).
inside(r1,o1).

semantic_region(f1).
belongs_to_detection(f1,o1).
region_role(f1,foreign_object).
region_centroid(f1,point(71,21)).
region_box(f1,box(52,6,89,40)).
region_area(f1,651).
semantic_support(f1,89).
feature_contrast(f1,7).
edge_strength(f1,17).
confidence(f1,70).
on_region(f1,r1).
inside(f1,o1).

semantic_region(f2).
belongs_to_detection(f2,o1).
region_role(f2,foreign_object).
region_centroid(f2,point(55,63)).
region_box(f2,box(17,18,92,97)).
region_area(f2,2071).
semantic_support(f2,90).
feature_contrast(f2,3).
edge_strength(f2,8).
confidence(f2,66).
on_region(f2,r1).
inside(f2,o1).

% Geometric Primitives

geometric_primitive(g1).
belongs_to_detection(g1,o1).
primitive_type(g1,ellipse).
primitive_box(g1,box(16,6,127,127)).
fit_error(g1,14).
edge_confidence(g1,3).
confidence(g1,86).
describes_region(g1,r1).
ellipse(g1,point(71,66),axes(105,125),angle(27)).

geometric_primitive(g2).
belongs_to_detection(g2,o1).
primitive_type(g2,curve_segment).
primitive_box(g2,box(25,1,100,99)).
length(g2,60).
orientation(g2,112).
fit_error(g2,28).
edge_confidence(g2,74).
semantic_support(g2,93).
confidence(g2,82).
location(g2,interior).
endpoints(g2,point(77,1),point(74,98)).

% Boundary Primitives

boundary_primitive(b1).
belongs_to_detection(b1,o1).
boundary_source(b1,ellipse).
from_ref(b1,g1).
primitive_type(b1,ellipse_perimeter).
primitive_box(b1,box(14,4,128,128)).
edge_confidence(b1,3).
semantic_support(b1,54).
confidence(b1,29).
inside(b1,o1).

boundary_primitive(b2).
belongs_to_detection(b2,o1).
boundary_source(b2,silhouette).
from_ref(b2,r1).
primitive_type(b2,silhouette).
primitive_box(b2,box(8,0,132,133)).
edge_confidence(b2,4).
semantic_support(b2,56).
confidence(b2,30).
inside(b2,o1).
length(b2,473).

% Relations
on_perimeter(b1,r1).
on_perimeter(b2,r1).
on_region(f1,r1).
on_region(f2,r1).


%%%%%


%% Candidate Primitives 

candidate_primitive(P) :- semantic_region(P).
candidate_primitive(P) :- geometric_primitive(P).
candidate_primitive(P) :- boundary_primitive(P).

%% Selection

{ explains(P,O) } :-
    candidate_primitive(P),
    belongs_to_detection(P,O).

% cardinality for semantic regions
L { explains(P,O) : semantic_region(P), belongs_to_detection(P,O) } U :- 
    detection(O), class(O,C),bound(C,semantic_region,L,U).

% ---------- Structural requirements ----------

included_body(O) :-
    explains(P,O),
    semantic_region(P),
    region_role(P,body).

:- detection(O), not included_body(O).


% ---------- Features ----------
feature(P, semantic_support, S) :-
    semantic_support(P,S).

feature(P, confidence, S) :-
    confidence(P,S).

feature(P, region_objectness, S) :-
    region_objectness(P,S).

feature(P, region_coherence, S) :-
    region_coherence(P,S).

feature(P, edge_confidence, S) :-
    edge_confidence(P,S).

feature(P, region_role(Role), 100) :-
    semantic_region(P),
    region_role(P,Role).

feature(P, primitive_type(Geom), 100) :-
    geometric_primitive(P),
    primitive_type(P,Geom).

feature(P, boundary_type(BType), 100) :-
    boundary_primitive(P),
    primitive_type(P,BType).

feature(P, boundary_source(Source), 100) :-
    boundary_primitive(P),
    boundary_source(P,Source).

feature(P, describes_role(Role), 100) :-
    geometric_primitive(P),
    describes_region(P,R),
    region_role(R,Role).

feature(P, on_region_role(Role), 100) :-
    on_region(P,R),
    region_role(R,Role).

feature(P, on_perimeter_role(Role), 100) :-
    on_perimeter(P,R),
    region_role(R,Role).


% ---------- Penalty features ----------

penalty_feature(P, fit_error, E) :-
    fit_error(P,E).

penalty_feature(P, complexity, 1) :-
    candidate_primitive(P).

penalty_feature(P, foreign_object, 50) :-
    semantic_region(P),
    region_role(P,foreign_object).

% boundary noise penalty
penalty_feature(P, weak_boundary, V) :-
    boundary_primitive(P),
    edge_confidence(P,E),
    V = 100 - E.


% ---------- Pairwise penalties ----------

pair_penalty(P1,P2,redundancy,V) :-
    explains(P1,O),
    explains(P2,O),
    overlap(P1,P2,V),
    P1 != P2.

#minimize { W*V@2,O,P1,P2,F :
    explains(P1,O),
    explains(P2,O),
    pair_penalty(P1,P2,F,V),
    class(O,C),
    weight(C,F,W)
}.

% ---------- Optimization ----------

#maximize { W*S@3,O,P,F :
    explains(P,O),
    feature(P,F,S),
    class(O,C),
    weight(C,F,W)
}.

#minimize { W*E@3,O,P,F :
    explains(P,O),
    penalty_feature(P,F,E),
    class(O,C),
    weight(C,F,W)
}.


%% Learned bounds and weights 

% ---------- Example bounds ----------

bound(animal_urchin,semantic_region,1,2).

% ---------- Example weights ----------

weight(animal_urchin,semantic_support,2).
weight(animal_urchin,confidence,1).
weight(animal_urchin,region_objectness,2).
weight(animal_urchin,region_coherence,1).
weight(animal_urchin,edge_confidence,2).

weight(animal_urchin,region_role(body),4).
weight(animal_urchin,region_role(foreign_object),-3).

weight(animal_urchin,primitive_type(ellipse),4).
weight(animal_urchin,primitive_type(line_segment),3).
weight(animal_urchin,primitive_type(curve_segment),1).

weight(animal_urchin,boundary_type(silhouette),2).
weight(animal_urchin,boundary_type(ellipse_perimeter),1).
weight(animal_urchin,boundary_source(silhouette),2).
weight(animal_urchin,boundary_source(ellipse),1).

weight(animal_urchin,describes_role(body),4).
weight(animal_urchin,on_perimeter_role(body),4).
weight(animal_urchin,on_region_role(body),2).

weight(animal_urchin,fit_error,1).
weight(animal_urchin,complexity,3).
weight(animal_urchin,foreign_object,3).
weight(animal_urchin,weak_boundary,1).
weight(animal_urchin,redundancy,2).

% ---------- Output ----------

#show explains/2.
%#show included_body/1.
%#show included_boundary/1.