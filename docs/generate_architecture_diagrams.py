#!/usr/bin/env python3
"""
Architecture Diagram Generator
Generates visual architecture diagrams for the Kotlin Development Tools project
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_system_architecture():
    """Create the main system architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Colors
    primary_color = '#2E86AB'
    secondary_color = '#A23B72'
    accent_color = '#F18F01'
    bg_color = '#F5F5F5'
    text_color = '#2C3E50'
    
    # Title
    ax.text(8, 11.5, 'Kotlin Development Tools - System Architecture', 
            fontsize=20, fontweight='bold', ha='center', color=text_color)
    
    # Main Application Layer
    app_box = FancyBboxPatch((1, 8.5), 14, 2.5, 
                             boxstyle="round,pad=0.1", 
                             facecolor=primary_color, 
                             edgecolor='black', 
                             alpha=0.8)
    ax.add_patch(app_box)
    ax.text(8, 10.2, 'Application Layer', fontsize=16, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # KDoc Module
    kdoc_box = FancyBboxPatch((2, 9), 3, 1.5, 
                              boxstyle="round,pad=0.05", 
                              facecolor='white', 
                              edgecolor=primary_color, 
                              linewidth=2)
    ax.add_patch(kdoc_box)
    ax.text(3.5, 9.75, 'KDoc Generation', fontsize=12, fontweight='bold', 
            ha='center', va='center', color=text_color)
    ax.text(3.5, 9.3, '• Kdoc.py\n• KdocGenerator.py', fontsize=10, 
            ha='center', va='center', color=text_color)
    
    # Test Generator Module
    test_box = FancyBboxPatch((11, 9), 3, 1.5, 
                              boxstyle="round,pad=0.05", 
                              facecolor='white', 
                              edgecolor=primary_color, 
                              linewidth=2)
    ax.add_patch(test_box)
    ax.text(12.5, 9.75, 'Test Generation', fontsize=12, fontweight='bold', 
            ha='center', va='center', color=text_color)
    ax.text(12.5, 9.3, '• TestCaseGenerator.py', fontsize=10, 
            ha='center', va='center', color=text_color)
    
    # Core Services Layer
    services_box = FancyBboxPatch((1, 5.5), 14, 2.5, 
                                  boxstyle="round,pad=0.1", 
                                  facecolor=secondary_color, 
                                  edgecolor='black', 
                                  alpha=0.8)
    ax.add_patch(services_box)
    ax.text(8, 7.2, 'Core Services Layer', fontsize=16, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # LLM Client
    llm_box = FancyBboxPatch((2, 6), 3, 1.5, 
                             boxstyle="round,pad=0.05", 
                             facecolor='white', 
                             edgecolor=secondary_color, 
                             linewidth=2)
    ax.add_patch(llm_box)
    ax.text(3.5, 6.75, 'LLM Client', fontsize=12, fontweight='bold', 
            ha='center', va='center', color=text_color)
    ax.text(3.5, 6.3, '• Ollama API\n• CodeLlama', fontsize=10, 
            ha='center', va='center', color=text_color)
    
    # Embedding Indexer
    embed_box = FancyBboxPatch((6.5, 6), 3, 1.5, 
                               boxstyle="round,pad=0.05", 
                               facecolor='white', 
                               edgecolor=secondary_color, 
                               linewidth=2)
    ax.add_patch(embed_box)
    ax.text(8, 6.75, 'Embedding Indexer', fontsize=12, fontweight='bold', 
            ha='center', va='center', color=text_color)
    ax.text(8, 6.3, '• CodeBERT\n• FAISS Index', fontsize=10, 
            ha='center', va='center', color=text_color)
    
    # Prompt Builder
    prompt_box = FancyBboxPatch((11, 6), 3, 1.5, 
                                boxstyle="round,pad=0.05", 
                                facecolor='white', 
                                edgecolor=secondary_color, 
                                linewidth=2)
    ax.add_patch(prompt_box)
    ax.text(12.5, 6.75, 'Prompt Builder', fontsize=12, fontweight='bold', 
            ha='center', va='center', color=text_color)
    ax.text(12.5, 6.3, '• Templates\n• Context', fontsize=10, 
            ha='center', va='center', color=text_color)
    
    # External Dependencies Layer
    deps_box = FancyBboxPatch((1, 2.5), 14, 2.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor=accent_color, 
                              edgecolor='black', 
                              alpha=0.8)
    ax.add_patch(deps_box)
    ax.text(8, 4.2, 'External Dependencies', fontsize=16, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Ollama
    ollama_box = FancyBboxPatch((2, 3), 3, 1.5, 
                                boxstyle="round,pad=0.05", 
                                facecolor='white', 
                                edgecolor=accent_color, 
                                linewidth=2)
    ax.add_patch(ollama_box)
    ax.text(3.5, 3.75, 'Ollama', fontsize=12, fontweight='bold', 
            ha='center', va='center', color=text_color)
    ax.text(3.5, 3.3, '• Local LLM\n• API Server', fontsize=10, 
            ha='center', va='center', color=text_color)
    
    # Transformers
    trans_box = FancyBboxPatch((6.5, 3), 3, 1.5, 
                               boxstyle="round,pad=0.05", 
                               facecolor='white', 
                               edgecolor=accent_color, 
                               linewidth=2)
    ax.add_patch(trans_box)
    ax.text(8, 3.75, 'Transformers', fontsize=12, fontweight='bold', 
            ha='center', va='center', color=text_color)
    ax.text(8, 3.3, '• CodeBERT\n• Tokenization', fontsize=10, 
            ha='center', va='center', color=text_color)
    
    # FAISS
    faiss_box = FancyBboxPatch((11, 3), 3, 1.5, 
                               boxstyle="round,pad=0.05", 
                               facecolor='white', 
                               edgecolor=accent_color, 
                               linewidth=2)
    ax.add_patch(faiss_box)
    ax.text(12.5, 3.75, 'FAISS', fontsize=12, fontweight='bold', 
            ha='center', va='center', color=text_color)
    ax.text(12.5, 3.3, '• Vector Search\n• Similarity Index', fontsize=10, 
            ha='center', va='center', color=text_color)
    
    # Data Flow Layer
    data_box = FancyBboxPatch((1, 0.5), 14, 1.5, 
                              boxstyle="round,pad=0.1", 
                              facecolor='#34495E', 
                              edgecolor='black', 
                              alpha=0.8)
    ax.add_patch(data_box)
    ax.text(8, 1.2, 'Data Layer', fontsize=16, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Input/Output
    ax.text(3, 0.8, 'Input: Kotlin Files (.kt)', fontsize=10, 
            ha='center', va='center', color='white')
    ax.text(8, 0.8, 'Processing: Embeddings & Indices', fontsize=10, 
            ha='center', va='center', color='white')
    ax.text(13, 0.8, 'Output: KDoc + Tests', fontsize=10, 
            ha='center', va='center', color='white')
    
    # Add arrows for data flow
    # App to Services
    arrow1 = ConnectionPatch((3.5, 9), (3.5, 7.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc=text_color, ec=text_color)
    ax.add_patch(arrow1)
    
    arrow2 = ConnectionPatch((12.5, 9), (8, 7.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc=text_color, ec=text_color)
    ax.add_patch(arrow2)
    
    # Services to Dependencies
    arrow3 = ConnectionPatch((3.5, 6), (3.5, 4.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc=text_color, ec=text_color)
    ax.add_patch(arrow3)
    
    arrow4 = ConnectionPatch((8, 6), (8, 4.5), "data", "data",
                            arrowstyle="->", shrinkA=5, shrinkB=5,
                            mutation_scale=20, fc=text_color, ec=text_color)
    ax.add_patch(arrow4)
    
    plt.tight_layout()
    plt.savefig('system_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_data_flow_diagram():
    """Create detailed data flow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    input_color = '#3498DB'
    process_color = '#E74C3C'
    output_color = '#2ECC71'
    
    # Title
    ax.text(7, 9.5, 'Data Flow Architecture', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Input
    input_box = FancyBboxPatch((1, 8), 2.5, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=input_color, 
                               edgecolor='black')
    ax.add_patch(input_box)
    ax.text(2.25, 8.5, 'Kotlin Files\n(.kt)', fontsize=11, fontweight='bold', 
            ha='center', va='center', color='white')
    
    # Processing steps
    steps = [
        (1, 6.5, 'File\nParser'),
        (4, 6.5, 'Class\nExtractor'),
        (7, 6.5, 'Embedding\nGenerator'),
        (10, 6.5, 'Semantic\nSearch'),
        (4, 4.5, 'Prompt\nBuilder'),
        (7, 4.5, 'LLM\nProcessor'),
        (10, 4.5, 'Code\nGenerator'),
        (7, 2.5, 'Quality\nChecker')
    ]
    
    for x, y, label in steps:
        box = FancyBboxPatch((x-0.75, y-0.4), 1.5, 0.8, 
                             boxstyle="round,pad=0.05", 
                             facecolor=process_color, 
                             edgecolor='black')
        ax.add_patch(box)
        ax.text(x, y, label, fontsize=10, fontweight='bold', 
                ha='center', va='center', color='white')
    
    # Outputs
    output_positions = [(2, 0.5, 'KDoc\nFiles'), (12, 0.5, 'Test\nFiles')]
    for x, y, label in output_positions:
        box = FancyBboxPatch((x-0.75, y-0.4), 1.5, 0.8, 
                             boxstyle="round,pad=0.1", 
                             facecolor=output_color, 
                             edgecolor='black')
        ax.add_patch(box)
        ax.text(x, y, label, fontsize=11, fontweight='bold', 
                ha='center', va='center', color='white')
    
    # Add flow arrows
    flows = [
        ((2.25, 8), (1.75, 6.9)),  # Input to parser
        ((1.75, 6.1), (3.25, 6.9)),  # Parser to extractor
        ((4.75, 6.5), (6.25, 6.5)),  # Extractor to embedding
        ((7.75, 6.5), (9.25, 6.5)),  # Embedding to search
        ((7, 6.1), (4.75, 4.9)),  # Search to prompt builder
        ((4.75, 4.5), (6.25, 4.5)),  # Prompt to LLM
        ((7.75, 4.5), (9.25, 4.5)),  # LLM to generator
        ((7, 4.1), (7, 2.9)),  # Generator to checker
        ((6.25, 2.5), (2.75, 0.9)),  # Checker to KDoc
        ((7.75, 2.5), (11.25, 0.9))  # Checker to Tests
    ]
    
    for start, end in flows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc='#34495E', ec='#34495E')
        ax.add_patch(arrow)
    
    plt.tight_layout()
    plt.savefig('data_flow_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_component_diagram():
    """Create detailed component interaction diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(6, 7.5, 'Component Interaction Diagram', 
            fontsize=16, fontweight='bold', ha='center')
    
    # Component positions and connections
    components = {
        'TestCaseGenerator': (6, 6, '#E74C3C'),
        'LLMClient': (2, 4, '#3498DB'),
        'EmbeddingIndexer': (6, 4, '#9B59B6'),
        'PromptBuilder': (10, 4, '#F39C12'),
        'KdocGenerator': (6, 2, '#2ECC71')
    }
    
    # Draw components
    for name, (x, y, color) in components.items():
        box = FancyBboxPatch((x-1, y-0.5), 2, 1, 
                             boxstyle="round,pad=0.1", 
                             facecolor=color, 
                             edgecolor='black',
                             alpha=0.8)
        ax.add_patch(box)
        ax.text(x, y, name, fontsize=10, fontweight='bold', 
                ha='center', va='center', color='white')
    
    # Add interaction arrows with labels
    interactions = [
        ((6, 5.5), (2, 4.5), 'generate()'),
        ((6, 5.5), (6, 4.5), 'retrieve_similar()'),
        ((6, 5.5), (10, 4.5), 'build_prompt()'),
        ((6, 5.5), (6, 2.5), 'uses'),
        ((7, 4), (9, 4), 'context'),
        ((5, 4), (3, 4), 'embeddings')
    ]
    
    for start, end, label in interactions:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=10, shrinkB=10,
                               mutation_scale=15, fc='#2C3E50', ec='#2C3E50')
        ax.add_patch(arrow)
        
        # Add label
        mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.2, label, fontsize=8, 
                ha='center', va='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('component_interaction.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating architecture diagrams...")
    create_system_architecture()
    create_data_flow_diagram()
    create_component_diagram()
    print("Architecture diagrams generated successfully!")
