// Interactive Model Explorer for R_V Research
// Compare R_V profiles across 6 validated architectures

class ModelExplorer {
    constructor() {
        this.container = document.getElementById('model-explorer');
        if (!this.container) return;
        
        this.canvas = document.getElementById('explorer-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        // Model data from paper
        this.models = {
            mistral: {
                name: 'Mistral-7B',
                layers: 32,
                criticalLayer: 27,
                criticalDepth: 0.84,
                contraction: -16.5,
                cohensD: -3.56,
                color: '#6366f1',
                description: 'Base model for main experiments. Strong recursive contraction effect.'
            },
            llama: {
                name: 'Llama-3.1-8B',
                layers: 32,
                criticalLayer: 27,
                criticalDepth: 0.84,
                contraction: -14.2,
                cohensD: -3.12,
                color: '#10b981',
                description: 'Meta\'s latest open model. Replicates effect at same relative depth.'
            },
            qwen: {
                name: 'Qwen-2-7B',
                layers: 28,
                criticalLayer: 24,
                criticalDepth: 0.86,
                contraction: -12.8,
                cohensD: -2.98,
                color: '#f59e0b',
                description: 'Alibaba\'s Qwen series. Higher critical depth (86%).'
            },
            phi3: {
                name: 'Phi-3-medium',
                layers: 32,
                criticalLayer: 27,
                criticalDepth: 0.84,
                contraction: -15.5,
                cohensD: -3.21,
                color: '#ec4899',
                description: 'Microsoft\'s Phi-3. Strong training efficiency, clear effect.'
            },
            gemma: {
                name: 'Gemma-2-9B',
                layers: 42,
                criticalLayer: 35,
                criticalDepth: 0.83,
                contraction: -18.3,
                cohensD: -3.45,
                color: '#8b5cf6',
                description: 'Google\'s Gemma-2. Largest contraction among dense models.'
            },
            mixtral: {
                name: 'Mixtral-8x7B',
                layers: 32,
                criticalLayer: 27,
                criticalDepth: 0.84,
                contraction: -24.3,
                cohensD: -4.21,
                color: '#06b6d4',
                description: 'MoE architecture. Amplified effect (d=-4.21). Routing specialization.'
            }
        };
        
        this.activeModels = new Set(['mistral', 'mixtral']); // Default selection
        this.viewMode = 'overlay'; // 'overlay' or 'normalized'
        this.hoveredModel = null;
        
        this.setupControls();
        this.setupInteraction();
        this.render();
    }
    
    setupControls() {
        const controls = document.getElementById('explorer-controls');
        if (!controls) return;
        
        // Model toggles
        const modelToggles = document.getElementById('model-toggles');
        if (modelToggles) {
            modelToggles.innerHTML = Object.entries(this.models).map(([key, model]) => `
                <label class="model-toggle" data-model="${key}">
                    <input type="checkbox" ${this.activeModels.has(key) ? 'checked' : ''}>
                    <span class="toggle-color" style="background: ${model.color}"></span>
                    <span class="toggle-name">${model.name}</span>
                    <span class="toggle-d">d=${model.cohensD.toFixed(2)}</span>
                </label>
            `).join('');
            
            // Add click handlers
            modelToggles.querySelectorAll('.model-toggle').forEach(toggle => {
                toggle.addEventListener('click', (e) => {
                    const modelKey = toggle.dataset.model;
                    const checkbox = toggle.querySelector('input');
                    
                    if (e.target !== checkbox) {
                        checkbox.checked = !checkbox.checked;
                    }
                    
                    if (checkbox.checked) {
                        this.activeModels.add(modelKey);
                    } else {
                        this.activeModels.delete(modelKey);
                    }
                    
                    this.render();
                    this.updateInfoPanel();
                });
            });
        }
        
        // View mode buttons
        const overlayBtn = document.getElementById('view-overlay');
        const normalizedBtn = document.getElementById('view-normalized');
        
        if (overlayBtn && normalizedBtn) {
            overlayBtn.addEventListener('click', () => {
                this.viewMode = 'overlay';
                overlayBtn.classList.add('active');
                normalizedBtn.classList.remove('active');
                this.render();
            });
            
            normalizedBtn.addEventListener('click', () => {
                this.viewMode = 'normalized';
                normalizedBtn.classList.add('active');
                overlayBtn.classList.remove('active');
                this.render();
            });
        }
        
        // Update initial info panel
        this.updateInfoPanel();
    }
    
    setupInteraction() {
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            // Check if hovering over a model line
            const hovered = this.getHoveredModel(x, y);
            if (hovered !== this.hoveredModel) {
                this.hoveredModel = hovered;
                this.canvas.style.cursor = hovered ? 'pointer' : 'default';
                this.render();
            }
        });
        
        this.canvas.addEventListener('click', (e) => {
            if (this.hoveredModel) {
                this.selectModel(this.hoveredModel);
            }
        });
        
        this.canvas.addEventListener('mouseleave', () => {
            this.hoveredModel = null;
            this.render();
        });
    }
    
    getHoveredModel(mouseX, mouseY) {
        const padding = { top: 40, right: 40, bottom: 60, left: 70 };
        const graphWidth = this.canvas.width - padding.left - padding.right;
        const graphHeight = this.canvas.height - padding.top - padding.bottom;
        
        // Check if within graph area
        if (mouseX < padding.left || mouseX > this.canvas.width - padding.right ||
            mouseY < padding.top || mouseY > this.canvas.height - padding.bottom) {
            return null;
        }
        
        // Find closest model line
        let closestModel = null;
        let closestDistance = Infinity;
        
        for (const modelKey of this.activeModels) {
            const model = this.models[modelKey];
            const criticalX = padding.left + model.criticalDepth * graphWidth;
            
            // Calculate distance to the contraction point
            const dx = mouseX - criticalX;
            const dy = mouseY - (padding.top + graphHeight * 0.3);
            const distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < 30 && distance < closestDistance) {
                closestDistance = distance;
                closestModel = modelKey;
            }
        }
        
        return closestModel;
    }
    
    selectModel(modelKey) {
        // Highlight in toggles
        document.querySelectorAll('.model-toggle').forEach(toggle => {
            toggle.classList.remove('selected');
            if (toggle.dataset.model === modelKey) {
                toggle.classList.add('selected');
            }
        });
        
        // Update info panel with detailed info
        const model = this.models[modelKey];
        const infoPanel = document.getElementById('model-info');
        if (infoPanel) {
            infoPanel.innerHTML = `
                <div class="info-header" style="border-left-color: ${model.color}">
                    <h4>${model.name}</h4>
                    <span class="info-badge">${model.layers} layers</span>
                </div>
                <div class="info-stats">
                    <div class="info-stat">
                        <span class="stat-label">Critical Layer</span>
                        <span class="stat-value">L${model.criticalLayer}</span>
                    </div>
                    <div class="info-stat">
                        <span class="stat-label">Depth</span>
                        <span class="stat-value">${(model.criticalDepth * 100).toFixed(0)}%</span>
                    </div>
                    <div class="info-stat">
                        <span class="stat-label">Contraction</span>
                        <span class="stat-value" style="color: ${model.contraction < -15 ? '#ef4444' : '#f59e0b'}">
                            ${model.contraction.toFixed(1)}%
                        </span>
                    </div>
                    <div class="info-stat">
                        <span class="stat-label">Effect Size</span>
                        <span class="stat-value" style="color: ${model.cohensD < -3.5 ? '#10b981' : '#6366f1'}">
                            d=${model.cohensD.toFixed(2)}
                        </span>
                    </div>
                </div>
                <p class="info-description">${model.description}</p>
            `;
        }
    }
    
    updateInfoPanel() {
        // Show summary if multiple selected, detail if one
        const selectedArray = Array.from(this.activeModels);
        const infoPanel = document.getElementById('model-info');
        
        if (!infoPanel) return;
        
        if (selectedArray.length === 0) {
            infoPanel.innerHTML = `
                <div class="info-placeholder">
                    <p>Select at least one model to view details</p>
                </div>
            `;
        } else if (selectedArray.length === 1) {
            this.selectModel(selectedArray[0]);
        } else {
            // Summary view
            const avgContraction = selectedArray.reduce((sum, k) => sum + this.models[k].contraction, 0) / selectedArray.length;
            const avgEffect = selectedArray.reduce((sum, k) => sum + this.models[k].cohensD, 0) / selectedArray.length;
            const strongest = selectedArray.reduce((best, k) => 
                this.models[k].cohensD < this.models[best].cohensD ? k : best
            );
            
            infoPanel.innerHTML = `
                <div class="info-header">
                    <h4>Model Comparison</h4>
                    <span class="info-badge">${selectedArray.length} models</span>
                </div>
                <div class="info-stats">
                    <div class="info-stat">
                        <span class="stat-label">Avg Contraction</span>
                        <span class="stat-value">${avgContraction.toFixed(1)}%</span>
                    </div>
                    <div class="info-stat">
                        <span class="stat-label">Avg Effect Size</span>
                        <span class="stat-value">d=${avgEffect.toFixed(2)}</span>
                    </div>
                    <div class="info-stat">
                        <span class="stat-label">Strongest Effect</span>
                        <span class="stat-value" style="color: ${this.models[strongest].color}">
                            ${this.models[strongest].name}
                        </span>
                    </div>
                </div>
                <p class="info-description">
                    All ${selectedArray.length} selected models show the characteristic geometric contraction 
                    signature at 78-86% network depth. Click a specific model line for detailed information.
                </p>
            `;
        }
    }
    
    // Generate R_V curve for a model
    generateCurve(model, numPoints = 100) {
        const points = [];
        const criticalDepth = model.criticalDepth;
        
        for (let i = 0; i <= numPoints; i++) {
            const depth = i / numPoints;
            let rv;
            
            if (this.viewMode === 'normalized') {
                // Normalized view: show contraction magnitude
                if (depth < criticalDepth - 0.1) {
                    rv = 1.0; // Baseline
                } else if (depth > criticalDepth + 0.1) {
                    rv = 1.0 + model.contraction / 100; // Contracted
                } else {
                    // Smooth transition
                    const t = (depth - (criticalDepth - 0.1)) / 0.2;
                    rv = 1.0 + (model.contraction / 100) * (0.5 - 0.5 * Math.cos(t * Math.PI));
                }
            } else {
                // Overlay view: absolute R_V values
                // Base curve decreases slightly with depth
                const baseRV = 1.0 - depth * 0.05;
                
                if (depth < criticalDepth - 0.05) {
                    rv = baseRV;
                } else if (depth > criticalDepth + 0.05) {
                    // Contracted region
                    const contractionFactor = 1 + model.contraction / 100;
                    rv = baseRV * (0.7 + 0.3 * contractionFactor);
                } else {
                    // Transition
                    const t = (depth - (criticalDepth - 0.05)) / 0.1;
                    const contractionFactor = 1 + model.contraction / 100;
                    const targetRV = baseRV * (0.7 + 0.3 * contractionFactor);
                    rv = baseRV + (targetRV - baseRV) * (0.5 - 0.5 * Math.cos(t * Math.PI));
                }
            }
            
            points.push({ depth, rv });
        }
        
        return points;
    }
    
    render() {
        const ctx = this.ctx;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        // Clear
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, width, height);
        
        const padding = { top: 40, right: 40, bottom: 60, left: 70 };
        const graphWidth = width - padding.left - padding.right;
        const graphHeight = height - padding.top - padding.bottom;
        
        // Draw grid
        this.drawGrid(ctx, padding, graphWidth, graphHeight);
        
        // Draw axes
        this.drawAxes(ctx, padding, graphWidth, graphHeight);
        
        // Draw model curves
        for (const modelKey of this.activeModels) {
            const model = this.models[modelKey];
            const isHovered = modelKey === this.hoveredModel;
            this.drawModelCurve(ctx, model, modelKey, padding, graphWidth, graphHeight, isHovered);
        }
        
        // Draw title
        ctx.fillStyle = '#f0f0f5';
        ctx.font = 'bold 16px Inter, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(
            this.viewMode === 'normalized' ? 'Normalized R_V Contraction' : 'R_V by Network Depth',
            padding.left, 25
        );
    }
    
    drawGrid(ctx, padding, graphWidth, graphHeight) {
        ctx.strokeStyle = '#1a1a24';
        ctx.lineWidth = 1;
        
        // Horizontal grid lines
        for (let i = 0; i <= 5; i++) {
            const y = padding.top + graphHeight * i / 5;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(padding.left + graphWidth, y);
            ctx.stroke();
        }
        
        // Vertical grid lines (every 20%)
        for (let i = 0; i <= 5; i++) {
            const x = padding.left + graphWidth * i / 5;
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, padding.top + graphHeight);
            ctx.stroke();
        }
    }
    
    drawAxes(ctx, padding, graphWidth, graphHeight) {
        ctx.strokeStyle = '#4a4a5a';
        ctx.lineWidth = 2;
        ctx.fillStyle = '#a0a0b0';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        
        // X axis
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top + graphHeight);
        ctx.lineTo(padding.left + graphWidth, padding.top + graphHeight);
        ctx.stroke();
        
        // Y axis
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, padding.top + graphHeight);
        ctx.stroke();
        
        // X labels
        ctx.fillText('Network Depth', padding.left + graphWidth / 2, padding.top + graphHeight + 45);
        
        for (let i = 0; i <= 5; i++) {
            const pct = i * 20;
            const x = padding.left + graphWidth * i / 5;
            ctx.fillText(`${pct}%`, x, padding.top + graphHeight + 20);
        }
        
        // Y label (rotated)
        ctx.save();
        ctx.translate(20, padding.top + graphHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText(this.viewMode === 'normalized' ? 'Normalized R_V' : 'Relative R_V', 0, 0);
        ctx.restore();
        
        // Y labels
        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const val = this.viewMode === 'normalized' 
                ? 1 - i * 0.1  // 1.0 to 0.5
                : 1 - i * 0.1; // Same scale for now
            const y = padding.top + graphHeight * i / 5;
            ctx.fillText(val.toFixed(1), padding.left - 10, y + 4);
        }
    }
    
    drawModelCurve(ctx, model, modelKey, padding, graphWidth, graphHeight, isHovered) {
        const curve = this.generateCurve(model);
        
        // Draw curve
        ctx.beginPath();
        ctx.strokeStyle = model.color;
        ctx.lineWidth = isHovered ? 4 : 2.5;
        
        curve.forEach((point, i) => {
            const x = padding.left + point.depth * graphWidth;
            const y = padding.top + graphHeight * (1 - (point.rv - 0.5) / 0.5);
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        ctx.stroke();
        
        // Draw critical point marker
        const criticalX = padding.left + model.criticalDepth * graphWidth;
        const criticalPoint = curve.find(p => Math.abs(p.depth - model.criticalDepth) < 0.01);
        
        if (criticalPoint) {
            const criticalY = padding.top + graphHeight * (1 - (criticalPoint.rv - 0.5) / 0.5);
            
            // Glow effect for hovered model
            if (isHovered) {
                ctx.beginPath();
                ctx.arc(criticalX, criticalY, 12, 0, Math.PI * 2);
                ctx.fillStyle = model.color + '40';
                ctx.fill();
            }
            
            // Main marker
            ctx.beginPath();
            ctx.arc(criticalX, criticalY, 6, 0, Math.PI * 2);
            ctx.fillStyle = model.color;
            ctx.fill();
            ctx.strokeStyle = '#0a0a0f';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Label for hovered model
            if (isHovered) {
                ctx.fillStyle = '#f0f0f5';
                ctx.font = 'bold 12px Inter, sans-serif';
                ctx.textAlign = 'left';
                ctx.fillText(model.name, criticalX + 15, criticalY - 5);
                ctx.font = '11px JetBrains Mono, monospace';
                ctx.fillStyle = '#a0a0b0';
                ctx.fillText(`L${model.criticalLayer} | d=${model.cohensD.toFixed(2)}`, criticalX + 15, criticalY + 10);
            }
        }
        
        // Draw layer boundary line for critical zone
        ctx.strokeStyle = model.color + '30';
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(criticalX, padding.top);
        ctx.lineTo(criticalX, padding.top + graphHeight);
        ctx.stroke();
        ctx.setLineDash([]);
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    new ModelExplorer();
});
