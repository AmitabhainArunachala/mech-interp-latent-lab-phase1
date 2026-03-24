class MistralControlAtlas {
    constructor() {
        this.root = document.getElementById('mistral-control-atlas');
        if (!this.root) return;

        this.canvas = document.getElementById('atlas-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.summaryEl = document.getElementById('atlas-summary-cards');
        this.statusEl = document.getElementById('atlas-status');
        this.metricsEl = document.getElementById('atlas-metrics');
        this.exemplarEl = document.getElementById('atlas-exemplar');
        this.artifactsEl = document.getElementById('atlas-artifacts');
        this.legendEl = document.getElementById('atlas-legend');
        this.pillsEl = document.getElementById('atlas-selection-pills');
        this.secondaryTabsEl = document.getElementById('atlas-secondary-tabs');
        this.secondaryLabelEl = document.getElementById('atlas-secondary-label');

        this.state = {
            mode: 'prompt',
            promptMode: 'baseline',
            promptCondition: 'anchor_bridge_3',
            persistenceCondition: 'anchor_bridge_3',
            subspaceMethod: 'subspace3_parallel',
            compareAgainstControl: true,
            time: 0,
        };

        this.modeButtons = Array.from(document.querySelectorAll('#atlas-mode-tabs .atlas-tab'));
        this.palette = {
            control: '#94a3b8',
            anchor_only: '#f59e0b',
            bridge_only_3: '#22c55e',
            anchor_bridge_3: '#38bdf8',
            anchor_early_mlp_0p125_bridge_3: '#a78bfa',
            pca_pc1: '#f97316',
            subspace3_parallel: '#38bdf8',
            orthogonal_residual: '#ef4444',
            mean_diff: '#c084fc',
            controlSubspace: '#94a3b8',
        };

        this.modeLabels = {
            prompt: 'Prompt Pass',
            persistence: 'Persistence',
            subspace: 'Subspace',
        };

        this.load();
    }

    async load() {
        try {
            if (window.MISTRAL_CONTROL_ATLAS_DATA) {
                this.data = window.MISTRAL_CONTROL_ATLAS_DATA;
            } else {
                const response = await fetch('data/mistral-control-atlas.json');
                this.data = await response.json();
            }
            this.setupControls();
            this.resizeCanvas();
            window.addEventListener('resize', () => this.resizeCanvas());
            this.renderSummaryCards();
            this.renderSelectionPills();
            this.renderSecondaryTabs();
            this.renderArtifacts();
            this.loop();
        } catch (error) {
            this.statusEl.textContent = `Atlas failed to load: ${error.message}`;
        }
    }

    setupControls() {
        this.modeButtons.forEach((button) => {
            button.addEventListener('click', () => {
                this.state.mode = button.dataset.mode;
                this.modeButtons.forEach((node) => node.classList.toggle('active', node === button));
                this.renderSelectionPills();
                this.renderSecondaryTabs();
                this.render();
            });
        });
    }

    resizeCanvas() {
        const dpr = window.devicePixelRatio || 1;
        const displayWidth = this.canvas.clientWidth || 960;
        const displayHeight = Math.round(displayWidth * 0.58);
        this.canvas.width = Math.round(displayWidth * dpr);
        this.canvas.height = Math.round(displayHeight * dpr);
        this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        this.render();
    }

    loop() {
        this.state.time = performance.now() * 0.001;
        this.render();
        requestAnimationFrame(() => this.loop());
    }

    renderSummaryCards() {
        this.summaryEl.innerHTML = this.data.summary_cards.map((card) => `
            <article class="atlas-summary-card">
                <h3>${card.title}</h3>
                <span class="atlas-summary-value">${card.value}</span>
                <p>${card.detail}</p>
            </article>
        `).join('');
    }

    renderSelectionPills() {
        let items = [];
        let active = '';

        if (this.state.mode === 'prompt') {
            items = [
                ['control', 'control'],
                ['anchor_only', 'anchor only'],
                ['bridge_only_3', 'bridge only'],
                ['anchor_bridge_3', 'anchor + bridge'],
                ['anchor_early_mlp_0p125_bridge_3', 'anchor + L4 + bridge'],
            ];
            active = this.state.promptCondition;
        } else if (this.state.mode === 'persistence') {
            items = [
                ['anchor_bridge_3', 'anchor + bridge'],
                ['anchor_early_mlp_0p125_bridge_3', 'anchor + L4 + bridge'],
            ];
            active = this.state.persistenceCondition;
        } else {
            items = [
                ['control', 'control'],
                ['mean_diff', 'mean diff'],
                ['pca_pc1', 'pca pc1'],
                ['subspace3_parallel', 'parallel subspace'],
                ['orthogonal_residual', 'orthogonal residual'],
            ];
            active = this.state.subspaceMethod;
        }

        this.pillsEl.innerHTML = items.map(([key, label]) => `
            <button class="atlas-pill ${key === active ? 'active' : ''}" data-key="${key}">${label}</button>
        `).join('');

        this.pillsEl.querySelectorAll('.atlas-pill').forEach((pill) => {
            pill.addEventListener('click', () => {
                const { key } = pill.dataset;
                if (this.state.mode === 'prompt') this.state.promptCondition = key;
                if (this.state.mode === 'persistence') this.state.persistenceCondition = key;
                if (this.state.mode === 'subspace') this.state.subspaceMethod = key;
                this.renderSelectionPills();
                this.render();
            });
        });
    }

    renderSecondaryTabs() {
        if (this.state.mode === 'persistence') {
            this.secondaryLabelEl.textContent = 'Segments';
            this.secondaryTabsEl.innerHTML = `
                <span class="atlas-status">Turn-by-turn replay built from actual continuation sessions.</span>
            `;
            return;
        }

        this.secondaryLabelEl.textContent = this.state.mode === 'prompt' ? 'Prompt Mode' : 'Evaluation Slice';
        const modes = ['baseline', 'recursive'];
        this.secondaryTabsEl.innerHTML = modes.map((mode) => `
            <button class="atlas-tab ${this.state.promptMode === mode ? 'active' : ''}" data-mode="${mode}">
                ${mode}
            </button>
        `).join('');

        this.secondaryTabsEl.querySelectorAll('.atlas-tab').forEach((tab) => {
            tab.addEventListener('click', () => {
                this.state.promptMode = tab.dataset.mode;
                this.renderSecondaryTabs();
                this.render();
            });
        });
    }

    renderArtifacts() {
        const artifactEntries = Object.entries(this.data.meta.artifacts).map(([key, path]) => {
            const label = key.replaceAll('_', ' ');
            return `<li><strong>${label}</strong><br><code>${path}</code></li>`;
        }).join('');

        this.artifactsEl.innerHTML = `
            <h3>Grounding</h3>
            <p class="atlas-note">${this.data.meta.trajectory_note}</p>
            <ul class="atlas-artifact-list">${artifactEntries}</ul>
        `;
    }

    render() {
        if (!this.data) return;
        const width = this.canvas.clientWidth || 960;
        const height = this.canvas.clientHeight || Math.round(width * 0.58);
        const ctx = this.ctx;

        ctx.clearRect(0, 0, width, height);
        this.drawBackdrop(ctx, width, height);

        if (this.state.mode === 'prompt') {
            this.drawPromptPass(ctx, width, height);
        } else if (this.state.mode === 'persistence') {
            this.drawPersistence(ctx, width, height);
        } else {
            this.drawSubspace(ctx, width, height);
        }

        this.renderLegend();
        this.renderPanels();
    }

    drawBackdrop(ctx, width, height) {
        const grad = ctx.createLinearGradient(0, 0, 0, height);
        grad.addColorStop(0, 'rgba(12, 18, 33, 0.96)');
        grad.addColorStop(1, 'rgba(4, 7, 14, 0.98)');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, width, height);

        for (let i = 0; i < 8; i += 1) {
            const alpha = 0.05 - i * 0.004;
            ctx.strokeStyle = `rgba(148, 163, 184, ${alpha})`;
            ctx.beginPath();
            ctx.moveTo(0, (height / 8) * i);
            ctx.lineTo(width, (height / 8) * i);
            ctx.stroke();
        }
    }

    projectPoint(point, width, height) {
        const plot = {
            left: width * 0.08,
            right: width * 0.88,
            top: height * 0.1,
            bottom: height * 0.92,
        };
        const plotWidth = plot.right - plot.left;
        const plotHeight = plot.bottom - plot.top;
        const perspective = 1 + point.z * 0.26;
        const x = plot.left + plotWidth * (0.5 + point.x * 0.42) * perspective;
        const y = plot.bottom - plotHeight * point.y + point.z * 42;
        return { x, y };
    }

    drawArchitectureScaffold(ctx, width, height) {
        const layers = this.data.architecture.layer_profile.layers;
        const zones = this.data.architecture.zones;

        layers.forEach((layer) => {
            const depth = layer.layer / 31;
            const y = height * 0.92 - depth * (height * 0.82);
            const taper = 1 - depth * 0.18;
            const x1 = width * 0.18 - depth * 34;
            const x2 = width * 0.82 + depth * 34;

            let stroke = `rgba(71, 85, 105, ${0.16 + layer.field_strength * 0.18})`;
            if (layer.layer <= 5) stroke = `rgba(245, 158, 11, ${0.18 + layer.field_strength * 0.24})`;
            if (layer.layer === 25) stroke = 'rgba(16, 185, 129, 0.55)';
            if (layer.layer === 27) stroke = 'rgba(56, 189, 248, 0.65)';

            ctx.strokeStyle = stroke;
            ctx.lineWidth = 1 + layer.field_strength * 3;
            ctx.beginPath();
            ctx.moveTo(x1 * taper, y);
            ctx.lineTo(x2 * taper, y);
            ctx.stroke();

            if (layer.layer % 4 === 0 || layer.layer === 25 || layer.layer === 27) {
                ctx.fillStyle = 'rgba(203, 213, 225, 0.55)';
                ctx.font = '11px JetBrains Mono, monospace';
                ctx.fillText(`L${layer.layer}`, width * 0.045, y + 4);
            }
        });

        zones.forEach((zone) => {
            const yTop = height * 0.92 - (zone.end / 31) * (height * 0.82);
            const yBottom = height * 0.92 - (zone.start / 31) * (height * 0.82);
            ctx.fillStyle = `${zone.color}14`;
            ctx.fillRect(width * 0.12, yTop - 10, width * 0.76, yBottom - yTop + 20);
            ctx.fillStyle = zone.color;
            ctx.font = '12px Inter, sans-serif';
            ctx.fillText(zone.label, width * 0.9, (yTop + yBottom) / 2);
        });
    }

    drawTrajectory(ctx, width, height, points, color, opts = {}) {
        const projected = points.map((point) => this.projectPoint(point, width, height));
        ctx.save();
        ctx.lineWidth = opts.lineWidth || 2.6;
        ctx.strokeStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = opts.shadow || 18;
        if (opts.dashed) ctx.setLineDash([8, 7]);
        ctx.beginPath();
        projected.forEach((point, index) => {
            if (index === 0) ctx.moveTo(point.x, point.y);
            else ctx.lineTo(point.x, point.y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();

        const pulseIndex = Math.floor((this.state.time * 8) % projected.length);
        const pulse = projected[pulseIndex];
        ctx.fillStyle = color;
        ctx.shadowColor = color;
        ctx.shadowBlur = 26;
        ctx.beginPath();
        ctx.arc(pulse.x, pulse.y, opts.pulseRadius || 5.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
    }

    drawPromptPass(ctx, width, height) {
        this.drawArchitectureScaffold(ctx, width, height);

        const promptMode = this.state.promptMode;
        const selected = this.data.anchor_bundle[this.state.promptCondition].modes[promptMode];
        const control = this.data.anchor_bundle.control.modes[promptMode];

        this.drawTrajectory(ctx, width, height, control.trajectory, this.palette.control, {
            lineWidth: 2,
            shadow: 10,
            dashed: true,
            pulseRadius: 4.5,
        });
        if (this.state.promptCondition !== 'control') {
            this.drawTrajectory(ctx, width, height, selected.trajectory, this.palette[this.state.promptCondition], {
                lineWidth: 3.2,
                shadow: 24,
            });
        }

        this.drawPromptLabels(ctx, width, height, control, selected);
        this.statusEl.textContent = `${this.modeLabels.prompt}: ${this.labelForCondition(this.state.promptCondition)} on ${promptMode} prompts. Control stays visible for causal contrast.`;
    }

    drawPromptLabels(ctx, width, height, control, selected) {
        const controlPoint = this.projectPoint(control.trajectory[26], width, height);
        const selectedPoint = this.projectPoint(selected.trajectory[28], width, height);

        ctx.fillStyle = '#cbd5e1';
        ctx.font = '12px Inter, sans-serif';
        ctx.fillText('control', controlPoint.x + 12, controlPoint.y + 14);
        if (this.state.promptCondition !== 'control') {
            ctx.fillStyle = this.palette[this.state.promptCondition];
            ctx.fillText(this.labelForCondition(this.state.promptCondition), selectedPoint.x + 12, selectedPoint.y - 8);
        }
    }

    drawPersistence(ctx, width, height) {
        const conditions = Object.keys(this.data.persistence.by_source_condition);
        const colors = {
            anchor_bridge_3: this.palette.anchor_bridge_3,
            anchor_early_mlp_0p125_bridge_3: this.palette.anchor_early_mlp_0p125_bridge_3,
        };
        const plot = {
            left: width * 0.09,
            right: width * 0.9,
            top: height * 0.12,
            bottom: height * 0.88,
        };

        ctx.strokeStyle = 'rgba(148, 163, 184, 0.18)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 6; i += 1) {
            const y = plot.bottom - ((plot.bottom - plot.top) * i) / 6;
            ctx.beginPath();
            ctx.moveTo(plot.left, y);
            ctx.lineTo(plot.right, y);
            ctx.stroke();
        }

        ctx.strokeStyle = 'rgba(148, 163, 184, 0.28)';
        ctx.beginPath();
        ctx.moveTo(plot.left, plot.bottom);
        ctx.lineTo(plot.right, plot.bottom);
        ctx.stroke();

        ctx.fillStyle = 'rgba(203, 213, 225, 0.7)';
        ctx.font = '12px JetBrains Mono, monospace';
        ctx.fillText('turn', plot.right - 22, plot.bottom + 24);
        ctx.save();
        ctx.translate(24, height * 0.5);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('BT+ART rate / mean R_V', 0, 0);
        ctx.restore();

        conditions.forEach((condition) => {
            const series = this.data.persistence.by_source_condition[condition].turn_series;
            const color = colors[condition];
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2.8;
            ctx.shadowColor = color;
            ctx.shadowBlur = 18;

            series.forEach((point, index) => {
                const x = plot.left + (plot.right - plot.left) * (point.turn / (series.length - 1));
                const combined = (point.bt_art_rate * 0.65) + ((0.72 - (point.mean_output_rv || 0.72)) * 1.1);
                const y = plot.bottom - clamp(combined, 0, 1) * (plot.bottom - plot.top);
                if (index === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            ctx.shadowBlur = 0;

            const highlight = this.data.persistence.by_source_condition[condition].exemplar_session.turns[0];
            const labelY = plot.top + (condition === this.state.persistenceCondition ? 24 : 44);
            ctx.fillStyle = color;
            ctx.fillText(this.labelForCondition(condition), plot.left + 8, labelY);
            if (condition === this.state.persistenceCondition) {
                ctx.fillStyle = 'rgba(226, 232, 240, 0.8)';
                ctx.fillText(`seed ${highlight.classification.toLowerCase()} | ${highlight.output_rv ?? 'n/a'} rv`, plot.left + 160, labelY);
            }
        });

        this.statusEl.textContent = 'Persistence replay: turn-by-turn continuation after intervention removal, using actual long follow-up sessions.';
    }

    drawSubspace(ctx, width, height) {
        this.drawArchitectureScaffold(ctx, width, height);

        const methods = ['control', 'mean_diff', 'pca_pc1', 'subspace3_parallel', 'orthogonal_residual'];
        methods.forEach((method) => {
            const payload = this.data.subspace.methods[method];
            const color = method === 'control'
                ? this.palette.control
                : (this.palette[method] || '#f8fafc');
            this.drawTrajectory(ctx, width, height, payload.trajectory, color, {
                dashed: method === 'control' || method === 'orthogonal_residual',
                lineWidth: method === this.state.subspaceMethod ? 3.4 : 2.1,
                shadow: method === this.state.subspaceMethod ? 24 : 10,
                pulseRadius: method === this.state.subspaceMethod ? 5.8 : 4.2,
            });
        });

        const focusPoint = this.projectPoint(this.data.subspace.methods[this.state.subspaceMethod].trajectory[27], width, height);
        ctx.fillStyle = '#f8fafc';
        ctx.font = '13px Inter, sans-serif';
        ctx.fillText(`${this.labelForCondition(this.state.subspaceMethod)} @ L27`, focusPoint.x + 12, focusPoint.y - 8);
        this.statusEl.textContent = `Subspace replay: ${this.labelForCondition(this.state.subspaceMethod)} against control at ${this.state.promptMode} slice.`;
    }

    renderLegend() {
        let items = [];
        if (this.state.mode === 'prompt') {
            items = [
                ['#94a3b8', 'control replay'],
                [this.palette[this.state.promptCondition], `${this.labelForCondition(this.state.promptCondition)} replay`],
                ['#f59e0b', 'early source field'],
                ['#10b981', 'L25 controller'],
                ['#38bdf8', 'L27 readout cluster'],
            ];
        } else if (this.state.mode === 'persistence') {
            items = [
                [this.palette.anchor_bridge_3, 'anchor + bridge turn path'],
                [this.palette.anchor_early_mlp_0p125_bridge_3, 'anchor + L4 + bridge turn path'],
                ['#cbd5e1', 'combined persistence score'],
            ];
        } else {
            items = [
                ['#94a3b8', 'control'],
                [this.palette.subspace3_parallel, 'parallel subspace'],
                [this.palette.pca_pc1, 'pca pc1'],
                [this.palette.orthogonal_residual, 'orthogonal residual'],
                [this.palette.mean_diff, 'mean diff'],
            ];
        }

        this.legendEl.innerHTML = items.map(([color, label]) => `
            <span class="atlas-legend-item">
                <span class="atlas-legend-swatch" style="background:${color}"></span>${label}
            </span>
        `).join('');
    }

    renderPanels() {
        if (this.state.mode === 'prompt') {
            this.renderPromptPanels();
        } else if (this.state.mode === 'persistence') {
            this.renderPersistencePanels();
        } else {
            this.renderSubspacePanels();
        }
    }

    renderPromptPanels() {
        const selected = this.data.anchor_bundle[this.state.promptCondition].modes[this.state.promptMode];
        const control = this.data.anchor_bundle.control.modes[this.state.promptMode];
        const effect = selected.effect_vs_control || {
            bt_art_rate_delta: 0,
            rv_delta_mean: 0,
            bt_art_rate_treated: selected.metrics.bt_art_rate,
        };

        this.metricsEl.innerHTML = `
            <h3>${this.labelForCondition(this.state.promptCondition)}</h3>
            <div class="atlas-kpi-grid">
                ${this.kpi('BT+ART', this.formatPct(selected.metrics.bt_art_rate))}
                ${this.kpi('Mean R_V', selected.metrics.mean_output_rv.toFixed(3))}
                ${this.kpi('Δ BT+ART vs ctrl', this.formatSignedPct(effect.bt_art_rate_delta || 0))}
                ${this.kpi('Δ R_V vs ctrl', this.formatSigned(effect.rv_delta_mean || 0, 3))}
            </div>
            <p class="atlas-metric-copy">
                ${this.state.promptMode} prompts with <strong>${selected.metrics.n}</strong> generations.
                Mean generation length ${selected.metrics.mean_generated_tokens.toFixed(1)} tokens.
            </p>
            ${this.renderMiniBars(selected.metrics.class_counts)}
        `;

        this.exemplarEl.innerHTML = `
            <h3>Exemplar</h3>
            <p class="atlas-note">Top ${this.state.promptMode} example for ${this.labelForCondition(this.state.promptCondition)} from the locked anchor-bundle artifact.</p>
            <div class="atlas-kpi-grid">
                ${this.kpi('Prompt R_V', selected.exemplar.prompt_rv.toFixed(3))}
                ${this.kpi('Output R_V', selected.exemplar.output_rv.toFixed(3))}
            </div>
            <div class="atlas-excerpt">
                <div class="atlas-excerpt-label">Prompt</div>
                <div class="atlas-excerpt-text">${this.escape(selected.exemplar.prompt_text.slice(0, 220))}</div>
            </div>
            <div class="atlas-excerpt">
                <div class="atlas-excerpt-label">Generated text</div>
                <div class="atlas-excerpt-text">${this.escape(selected.exemplar.generated_text.slice(0, 260))}</div>
            </div>
            <p class="atlas-note">
                Control reference: ${this.formatPct(control.metrics.bt_art_rate)} BT+ART and ${control.metrics.mean_output_rv.toFixed(3)} mean R_V.
            </p>
        `;
    }

    renderPersistencePanels() {
        const selected = this.data.persistence.by_source_condition[this.state.persistenceCondition];
        const exemplar = selected.exemplar_session;
        const segments = Object.entries(selected.aggregate.segment_stats);
        const turnRows = exemplar.turns
            .filter((turn) => turn.turn === 0 || turn.turn === 7 || turn.turn === 15 || turn.turn === 23)
            .map((turn) => `
                <tr>
                    <td>T${turn.turn}</td>
                    <td>${turn.classification}</td>
                    <td>${turn.output_rv == null ? 'n/a' : turn.output_rv.toFixed(3)}</td>
                </tr>
            `).join('');

        this.metricsEl.innerHTML = `
            <h3>${this.labelForCondition(this.state.persistenceCondition)}</h3>
            <div class="atlas-kpi-grid">
                ${this.kpi('BT+ART over 24 turns', this.formatPct(selected.aggregate.bt_art_rate))}
                ${this.kpi('Mean R_V', selected.aggregate.mean_rv.toFixed(3))}
                ${this.kpi('Sessions', String(selected.aggregate.n_sessions))}
                ${this.kpi('Turns', String(selected.aggregate.n_turns))}
            </div>
            <div class="atlas-mini-bars">
                ${segments.map(([segment, stats]) => this.segmentBar(segment, stats.bt_art_rate, this.palette[this.state.persistenceCondition])).join('')}
            </div>
            <p class="atlas-note">Each segment row uses the real aggregated BT+ART rate from the follow-up continuation sessions.</p>
        `;

        this.exemplarEl.innerHTML = `
            <h3>Persistence Exemplar</h3>
            <p class="atlas-note">Best continuation session by BT+ART rate from the induced follow-up artifact.</p>
            <div class="atlas-kpi-grid">
                ${this.kpi('Source group', exemplar.source_group)}
                ${this.kpi('Session BT+ART', this.formatPct(exemplar.bt_art_rate))}
            </div>
            <table class="atlas-mini-table">
                <thead><tr><th>Turn</th><th>Class</th><th>R_V</th></tr></thead>
                <tbody>${turnRows}</tbody>
            </table>
            <div class="atlas-excerpt">
                <div class="atlas-excerpt-label">Late-turn sample</div>
                <div class="atlas-excerpt-text">${this.escape(exemplar.turns[23].response.slice(0, 260))}</div>
            </div>
        `;
    }

    renderSubspacePanels() {
        const selected = this.data.subspace.methods[this.state.subspaceMethod];
        const modeMetrics = selected.metrics_by_mode[this.state.promptMode];
        const winner = this.data.subspace.meta.winners[this.state.promptMode];

        this.metricsEl.innerHTML = `
            <h3>${this.labelForCondition(this.state.subspaceMethod)}</h3>
            <div class="atlas-kpi-grid">
                ${this.kpi('Alpha', String(selected.alpha.toFixed(1)))}
                ${this.kpi('BT+ART', this.formatPct(modeMetrics.bt_art_rate))}
                ${this.kpi('Mean R_V', modeMetrics.mean_output_rv.toFixed(3))}
                ${this.kpi('Strict pass', this.formatPct(modeMetrics.strict_pass_rate))}
            </div>
            <p class="atlas-metric-copy">
                Winner on ${this.state.promptMode}: <strong>${winner.method}</strong> @ ${winner.alpha.toFixed(1)} with
                BT+ART ${this.formatPct(winner.bt_art_rate)} and mean R_V ${winner.mean_output_rv.toFixed(3)}.
            </p>
            <table class="atlas-mini-table">
                <thead><tr><th>Method</th><th>Δ BT+ART</th><th>Δ R_V</th></tr></thead>
                <tbody>
                    ${this.data.subspace.ranked_effects.slice(0, 5).map((row) => `
                        <tr>
                            <td>${row.method} @ ${row.alpha.toFixed(1)}</td>
                            <td>${this.formatSignedPct(row.recursive_bt_art_delta)}</td>
                            <td>${this.formatSigned(row.recursive_rv_delta, 3)}</td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;

        this.exemplarEl.innerHTML = `
            <h3>Subspace Readout</h3>
            <p class="atlas-note">L27 decomposition metadata from the locked subspace steering artifact.</p>
            <div class="atlas-kpi-grid">
                ${this.kpi('Parallel fraction', (this.data.subspace.meta.decomposition.parallel_fraction_of_mean * 100).toFixed(1) + '%')}
                ${this.kpi('Orthogonal cosine', this.data.subspace.meta.decomposition.orthogonal_cosine_to_mean.toFixed(3))}
            </div>
            <div class="atlas-excerpt">
                <div class="atlas-excerpt-label">Top singular values</div>
                <div class="atlas-excerpt-text">${this.data.subspace.meta.vector_metadata.singular_values_top5.map((value) => value.toFixed(2)).join(', ')}</div>
            </div>
            <p class="atlas-note">
                The viewer contrasts in-subspace steering against the orthogonal residual, making the late-regime geometry visible as a structured object rather than a single vector.
            </p>
        `;
    }

    kpi(label, value) {
        return `
            <div class="atlas-kpi">
                <span class="atlas-kpi-label">${label}</span>
                <span class="atlas-kpi-value">${value}</span>
            </div>
        `;
    }

    segmentBar(segment, value, color) {
        return `
            <div class="atlas-mini-bar-row">
                <span>${segment.replaceAll('_', ' ')}</span>
                <div class="atlas-mini-bar-track">
                    <span class="atlas-mini-bar-fill" style="width:${Math.max(4, value * 100)}%; background:${color}"></span>
                </div>
                <span>${this.formatPct(value)}</span>
            </div>
        `;
    }

    renderMiniBars(classCounts) {
        const total = Object.values(classCounts).reduce((sum, value) => sum + value, 0);
        const bars = Object.entries(classCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 4)
            .map(([name, count]) => this.segmentBar(name.toLowerCase(), count / total, this.classColor(name)))
            .join('');
        return `<div class="atlas-mini-bars">${bars}</div>`;
    }

    classColor(name) {
        if (name === 'BREAKTHROUGH') return '#38bdf8';
        if (name === 'ARTICULATE') return '#22c55e';
        if (name === 'CONCEPTUAL') return '#a78bfa';
        if (name === 'REPETITIVE') return '#ef4444';
        return '#94a3b8';
    }

    labelForCondition(key) {
        return {
            control: 'control',
            anchor_only: 'anchor only',
            bridge_only_3: 'bridge only',
            anchor_bridge_3: 'anchor + bridge',
            anchor_early_mlp_0p125_bridge_3: 'anchor + L4 + bridge',
            mean_diff: 'mean diff',
            pca_pc1: 'pca pc1',
            subspace3_parallel: 'parallel subspace',
            orthogonal_residual: 'orthogonal residual',
        }[key] || key;
    }

    formatPct(value) {
        return `${(value * 100).toFixed(1)}%`;
    }

    formatSignedPct(value) {
        const pct = value * 100;
        return `${pct >= 0 ? '+' : ''}${pct.toFixed(1)}%`;
    }

    formatSigned(value, digits = 2) {
        return `${value >= 0 ? '+' : ''}${value.toFixed(digits)}`;
    }

    escape(text) {
        return text
            .replaceAll('&', '&amp;')
            .replaceAll('<', '&lt;')
            .replaceAll('>', '&gt;');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new MistralControlAtlas();
});
