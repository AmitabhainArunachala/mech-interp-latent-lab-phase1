// R_V Interactive Calculator
// Demonstrates the geometric contraction metric in-browser

class RVCalculator {
    constructor() {
        this.canvas = document.getElementById('rv-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.animationId = null;
        
        // Default parameters
        this.params = {
            recursiveIntensity: 0.5,  // 0-1 slider
            layerDepth: 0.84,          // Normalized layer (L27/32 = 0.84)
            noiseLevel: 0.1
        };
        
        this.setupControls();
        this.render();
    }
    
    setupControls() {
        const controls = document.getElementById('rv-controls');
        if (!controls) return;
        
        // Recursive intensity slider
        const intensitySlider = document.getElementById('intensity-slider');
        const intensityValue = document.getElementById('intensity-value');
        if (intensitySlider) {
            intensitySlider.addEventListener('input', (e) => {
                this.params.recursiveIntensity = parseFloat(e.target.value);
                if (intensityValue) {
                    intensityValue.textContent = (this.params.recursiveIntensity * 100).toFixed(0) + '%';
                }
                this.render();
            });
        }
        
        // Prompt type buttons
        const baselineBtn = document.getElementById('baseline-btn');
        const recursiveBtn = document.getElementById('recursive-btn');
        
        if (baselineBtn) {
            baselineBtn.addEventListener('click', () => {
                this.params.recursiveIntensity = 0;
                if (intensitySlider) intensitySlider.value = 0;
                if (intensityValue) intensityValue.textContent = '0%';
                this.render();
            });
        }
        
        if (recursiveBtn) {
            recursiveBtn.addEventListener('click', () => {
                this.params.recursiveIntensity = 1;
                if (intensitySlider) intensitySlider.value = 1;
                if (intensityValue) intensityValue.textContent = '100%';
                this.render();
            });
        }
    }
    
    // Simulate participation ratio based on recursive intensity
    computePR(layer, recursive) {
        // Base PR decreases with depth
        const basePR = 100 - (layer * 30);
        
        // Recursive prompts show contraction in late layers
        if (layer > 0.75 && recursive > 0) {
            const contractionFactor = 1 - (recursive * 0.266 * (layer - 0.75) / 0.25);
            return basePR * contractionFactor;
        }
        
        return basePR + (Math.random() - 0.5) * this.params.noiseLevel * 10;
    }
    
    // Compute R_V = PR(L27) / PR(L5)
    computeRV() {
        const pr_early = this.computePR(0.15, 0); // L5/32 ≈ 0.15
        const pr_late = this.computePR(this.params.layerDepth, this.params.recursiveIntensity);
        return pr_late / pr_early;
    }
    
    render() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const ctx = this.ctx;
        
        // Clear
        ctx.fillStyle = '#0a0a0f';
        ctx.fillRect(0, 0, width, height);
        
        // Draw layer profile
        this.drawLayerProfile(ctx, width, height);
        
        // Draw R_V result
        this.drawRVResult(ctx, width, height);
        
        // Draw spheres visualization
        this.drawSpheres(ctx, width, height);
    }
    
    drawLayerProfile(ctx, width, height) {
        const padding = 60;
        const graphWidth = width * 0.55 - padding * 2;
        const graphHeight = height - padding * 2;
        
        ctx.save();
        ctx.translate(padding, padding);
        
        // Axes
        ctx.strokeStyle = '#2a2a35';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, graphHeight);
        ctx.lineTo(graphWidth, graphHeight);
        ctx.moveTo(0, 0);
        ctx.lineTo(0, graphHeight);
        ctx.stroke();
        
        // Axis labels
        ctx.fillStyle = '#a0a0b0';
        ctx.font = '12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Layer Depth', graphWidth / 2, graphHeight + 35);
        
        ctx.save();
        ctx.translate(-35, graphHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Participation Ratio', 0, 0);
        ctx.restore();
        
        // Grid lines
        ctx.strokeStyle = '#1a1a24';
        for (let i = 1; i < 5; i++) {
            ctx.beginPath();
            ctx.moveTo(0, graphHeight * i / 5);
            ctx.lineTo(graphWidth, graphHeight * i / 5);
            ctx.stroke();
        }
        
        // Layer markers
        ctx.fillStyle = '#606070';
        ctx.font = '10px JetBrains Mono, monospace';
        const layers = [5, 10, 15, 20, 25, 27, 32];
        layers.forEach(l => {
            const x = (l / 32) * graphWidth;
            ctx.fillText(`L${l}`, x, graphHeight + 18);
        });
        
        // Draw baseline curve
        ctx.beginPath();
        ctx.strokeStyle = '#4a4a5a';
        ctx.lineWidth = 2;
        for (let i = 0; i <= 32; i++) {
            const x = (i / 32) * graphWidth;
            const pr = this.computePR(i / 32, 0);
            const y = graphHeight - (pr / 100) * graphHeight;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
        
        // Draw recursive curve
        if (this.params.recursiveIntensity > 0) {
            ctx.beginPath();
            ctx.strokeStyle = '#6366f1';
            ctx.lineWidth = 3;
            for (let i = 0; i <= 32; i++) {
                const x = (i / 32) * graphWidth;
                const pr = this.computePR(i / 32, this.params.recursiveIntensity);
                const y = graphHeight - (pr / 100) * graphHeight;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }
            ctx.stroke();
        }
        
        // Highlight contraction zone
        if (this.params.recursiveIntensity > 0) {
            ctx.fillStyle = 'rgba(99, 102, 241, 0.1)';
            ctx.fillRect(0.75 * graphWidth, 0, 0.25 * graphWidth, graphHeight);
            
            ctx.fillStyle = '#818cf8';
            ctx.font = '11px Inter, sans-serif';
            ctx.fillText('Contraction Zone', 0.875 * graphWidth, 20);
        }
        
        // Legend
        ctx.font = '11px Inter, sans-serif';
        ctx.fillStyle = '#4a4a5a';
        ctx.fillText('● Baseline', 10, 20);
        if (this.params.recursiveIntensity > 0) {
            ctx.fillStyle = '#6366f1';
            ctx.fillText('● Recursive', 80, 20);
        }
        
        ctx.restore();
    }
    
    drawRVResult(ctx, width, height) {
        const rv = this.computeRV();
        const x = width * 0.75;
        const y = 80;
        
        // R_V value
        ctx.fillStyle = '#f0f0f5';
        ctx.font = 'bold 48px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(rv.toFixed(3), x, y);
        
        // Label
        ctx.fillStyle = '#a0a0b0';
        ctx.font = '14px Inter, sans-serif';
        ctx.fillText('R_V = PR(L27) / PR(L5)', x, y + 30);
        
        // Interpretation
        ctx.font = '16px Inter, sans-serif';
        if (rv < 0.95) {
            ctx.fillStyle = '#10b981';
            ctx.fillText('Geometric Contraction Detected', x, y + 60);
        } else if (rv > 1.05) {
            ctx.fillStyle = '#f59e0b';
            ctx.fillText('Geometric Expansion', x, y + 60);
        } else {
            ctx.fillStyle = '#a0a0b0';
            ctx.fillText('Near Baseline', x, y + 60);
        }
        
        // Effect size estimate
        const d = (1 - rv) / 0.075; // Approximate Cohen's d
        ctx.font = '12px JetBrains Mono, monospace';
        ctx.fillStyle = '#606070';
        ctx.fillText(`Cohen's d ≈ ${d.toFixed(2)}`, x, y + 85);
    }
    
    drawSpheres(ctx, width, height) {
        const centerX = width * 0.75;
        const baseY = height - 120;
        const rv = this.computeRV();
        
        // Baseline sphere
        const baseRadius = 45;
        ctx.beginPath();
        ctx.arc(centerX - 80, baseY, baseRadius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(74, 74, 90, 0.5)';
        ctx.fill();
        ctx.strokeStyle = '#4a4a5a';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        ctx.fillStyle = '#a0a0b0';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Early Layer', centerX - 80, baseY + 65);
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.fillText('PR = 1.0', centerX - 80, baseY + 80);
        
        // Arrow
        ctx.beginPath();
        ctx.moveTo(centerX - 30, baseY);
        ctx.lineTo(centerX + 30, baseY);
        ctx.strokeStyle = '#606070';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(centerX + 30, baseY);
        ctx.lineTo(centerX + 20, baseY - 5);
        ctx.lineTo(centerX + 20, baseY + 5);
        ctx.closePath();
        ctx.fillStyle = '#606070';
        ctx.fill();
        
        // Late layer sphere (contracted)
        const lateRadius = baseRadius * Math.sqrt(rv);
        ctx.beginPath();
        ctx.arc(centerX + 80, baseY, lateRadius, 0, Math.PI * 2);
        
        if (rv < 0.95) {
            ctx.fillStyle = 'rgba(99, 102, 241, 0.4)';
            ctx.strokeStyle = '#6366f1';
        } else {
            ctx.fillStyle = 'rgba(74, 74, 90, 0.5)';
            ctx.strokeStyle = '#4a4a5a';
        }
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.stroke();
        
        ctx.fillStyle = '#a0a0b0';
        ctx.font = '11px Inter, sans-serif';
        ctx.fillText('Late Layer (L27)', centerX + 80, baseY + 65);
        ctx.font = '10px JetBrains Mono, monospace';
        ctx.fillText(`PR = ${rv.toFixed(2)}`, centerX + 80, baseY + 80);
    }
}

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('rv-calculator');
    if (container) {
        new RVCalculator();
    }
});
