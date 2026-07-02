# Refining Underwater Object Detections through Visual Decomposition and Explanation

## Project Overview
This project focuses on underwater object detection, using **Seaclear**, **UOD**, and **RUOD** Datasets. The core model is **DEIMv2**, enhanced with a decomposition and explanation pipeline for improved bounding box accuracy in challenging underwater environments.

The assumption is that identifying the structural elements (atoms) of objects helps in finding better-aligned bounding boxes. By "explaining" the object through its primitives (geometric forms, contours, etc.), we can refine the detection to better match ground truth.

## Pipeline Structure
The system follows a modular pipeline: **DINOv3-DETR → Atom Extraction → Symbolic Reasoning → Explanation Refinement → Final Output**.

### 1. Object Detection
- **Input**: Raw underwater image.
- **Model**: DINOv3-DETR
- **Output**: Initial detected bounding boxes with class and confidence scores (e.g., "sea urchin 0.87").
- Crops the detected box with a margin for further processing.

### 2. Foundation Features (DINOv3)
- Extracts DINOv3 patch embeddings (H × W × C).
- Computes **Object Embedding** via average pooling.
- Generates:
  - **Object Support Map (S)**: Highlights regions semantically similar to the object (cosine similarity).
  - **Feature-Boundary Map (B_dino)**: Emphasizes boundaries and structures.
- Provides DINO Guidance for semantic regions, objectness, and boundary priors.

### 3. Primitive / Atom Extraction (Multi-Source)
- **A) Semantic Regions**: From DINO support map, fits primitives (e.g., ellipses for body) to high-support regions.
- **B) Learned Primitive Detector (PiDiNet)**: Multi-head detector on RGB crop + fused DINO features. Outputs probability maps for boundaries, lines, curves, junctions, elliptic regions.
- **C) Primitive Fitting**: 
- **D) Visual Atoms (Candidates)**:
- Extracts tokens/patches inside the crop and scores them for object support.
- Clusters supported tokens into region atoms.
- Computes geometric descriptors.
- Runs edge extractor for contour atoms.
- Computes atom relations.

### 4. Symbolic Reasoning (ASP)
- **Abductive Explanation**: Uses Answer Set Programming (ASP) solver to select a coherent set of atoms that best explains the object.
- **Knowledge Bases**:
  - Object templates (e.g., sea_urchin)
  - Geometric relations and constraints
  - Preferences/weights
- Enforces high-level relations:
  - Has body (ellipse)
  - Spike radial to body
  - Points outward
  - Coverage of body
  - Minimal complexity
- Outputs a selected explanation (e.g., body + radial spikes).

### 5. Explanation-Conditioned Refinement
- **Inputs**:
  - DINOv3 features (crop)
  - Atom maps (body/spikes/boundaries)
  - Original DETR box proposal
- **Refinement Head** (CNN/Transformer): Predicts box offsets or refined mask.
- Produces **Refined Box** and optional refined mask.

### 6. Final Output
- **Refined Bounding Box** with improved confidence (e.g., "sea urchin 0.92").
- **Benefits**:
  - More accurate localization
  - Explainable atoms
  - Interpretable reasoning
  - Robust to underwater noise

## Minimal Implementation Order
1. Crop DETR box + margin.
2. Extract DINO token patches inside crop.
3. Score tokens for object support.
4. Cluster supported tokens into region atoms.
5. Compute geometric descriptors for each region atom.
6. Run edge extraction for contour atoms.
7. Compute atom relations.
8. Export ASP facts.
9. Visualize atoms over the image.
10. Only then connect to ASP reasoning.

## Datasets
- **Seaclear**
- **UOD**
- **RUOD**

## Model
- **DINOv3-DETR** (base)

## Overall Idea
Foundation features provide semantic understanding; modern primitive extraction yields geometric atoms; ASP selects a coherent explanation; a neural regressor uses this explanation and features to refine localization.
