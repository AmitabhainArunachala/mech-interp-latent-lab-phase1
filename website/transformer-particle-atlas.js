/**
 * R_V Particle Atlas — Mistral-7B-v0.1
 *
 * Three.js WebGL particle visualization of a 32-layer Mistral transformer.
 * Maps mechanistic interpretability data (R_V contraction, field strength,
 * trajectory divergence) onto a 5D particle system.
 *
 * Data source: window.MISTRAL_CONTROL_ATLAS_DATA
 */

(function () {
  'use strict';

  // ─────────────────────────────────────────────
  // Constants
  // ─────────────────────────────────────────────
  var TOTAL_PARTICLES = 10240;
  var PARTICLES_PER_LAYER = Math.floor(TOTAL_PARTICLES / 32); // 320
  var SCENE_HEIGHT = 20;          // vertical extent (y-axis) for 32 layers
  var LAYER_SPACING = SCENE_HEIGHT / 31;
  var PLANE_RADIUS = 5.0;        // radius of each layer disc
  var POSITION_SCALE = 6.0;      // scale factor for trajectory x/z to world coords
  var TRANSITION_DURATION = 2.0;  // seconds for mode morph
  var TRAIL_LENGTH = 6;           // number of trail echo frames
  var BROWNIAN_STRENGTH = 0.015;
  var DRIFT_SPEED = 0.08;

  // Color palette
  var COLOR_BASELINE = new THREE.Color(0x4488ff);
  var COLOR_RECURSIVE_COOL = new THREE.Color(0xffaa22);
  var COLOR_RECURSIVE_HOT = new THREE.Color(0xfff0d0);
  var COLOR_ZONE_EARLY = new THREE.Color(0xf59e0b);
  var COLOR_ZONE_CONTROLLER = new THREE.Color(0x10b981);
  var COLOR_ZONE_READOUT = new THREE.Color(0x38bdf8);
  var COLOR_LAYER_DEFAULT = new THREE.Color(0x1a1a28);

  // ─────────────────────────────────────────────
  // Main class
  // ─────────────────────────────────────────────
  function TransformerParticleAtlas(canvas, data) {
    this.canvas = canvas;
    this.data = data;
    this.layers = data.architecture.layer_profile.layers;
    this.zones = data.architecture.zones;
    this.controlModes = data.anchor_bundle.control.modes;

    this.currentMode = 'baseline';
    this.targetMode = 'baseline';
    this.transitionProgress = 1.0;  // 1.0 = fully arrived
    this.isTransitioning = false;
    this.transitionDirection = 1;   // 1 = forward, for ping-pong

    this.clock = new THREE.Clock();
    this.mouse = new THREE.Vector2(-999, -999);
    this.raycaster = new THREE.Raycaster();
    this.hoveredLayer = -1;
    this.focusedLayer = -1;
    this.idleTime = 0;
    this.isUserInteracting = false;
    this.autoRotateEnabled = true;
    this.showTrajectories = true;
    this.showTrails = false;
    this.pendingMode = 'baseline';

    // Store computed positions
    this.baselinePositions = null;  // Float32Array
    this.recursivePositions = null; // Float32Array
    this.baselineColors = null;
    this.recursiveColors = null;
    this.baselineSizes = null;
    this.recursiveSizes = null;

    // Trail system
    this.trailHistory = [];
    this.trailMeshes = [];

    // Layer plane meshes for raycasting
    this.layerPlanes = [];
    this.layerLabels = [];

    this.init();
  }

  // ─────────────────────────────────────────────
  // Initialization
  // ─────────────────────────────────────────────
  TransformerParticleAtlas.prototype.init = function () {
    this.initScene();
    this.initStarField();
    this.initLayers();
    this.computeParticlePositions();
    this.initParticles();
    this.initTrajectories();
    this.initTrailSystem();
    this.initPostProcessing();
    this.initHUD();
    this.initInteraction();

    // Hide loading overlay
    var overlay = document.getElementById('loading-overlay');
    if (overlay) {
      setTimeout(function () {
        overlay.classList.add('hidden');
      }, 600);
    }

    this.animate();
  };

  TransformerParticleAtlas.prototype.initScene = function () {
    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      canvas: this.canvas,
      antialias: true,
      alpha: false,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(window.innerWidth, window.innerHeight);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x0a0a0f, 1);
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;

    // Handle WebGL context loss/restore
    var self = this;
    this.canvas.addEventListener('webglcontextlost', function (e) {
      e.preventDefault();
      console.warn('WebGL context lost — will attempt restore');
      self._contextLost = true;
    });
    this.canvas.addEventListener('webglcontextrestored', function () {
      console.log('WebGL context restored — reinitializing');
      self._contextLost = false;
      self.initPostProcessing();
    });

    // Scene
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x0a0a0f, 0.012);

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      55,
      window.innerWidth / window.innerHeight,
      0.1,
      200
    );
    this.camera.position.set(12, 10, 16);
    this.camera.lookAt(0, SCENE_HEIGHT * 0.45, 0);

    // Controls
    this.controls = new THREE.OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.target.set(0, SCENE_HEIGHT * 0.45, 0);
    this.controls.minDistance = 5;
    this.controls.maxDistance = 60;
    this.controls.autoRotate = true;
    this.controls.autoRotateSpeed = 0.3;
    this.controls.maxPolarAngle = Math.PI * 0.85;

    // Ambient light (for layer planes)
    var ambient = new THREE.AmbientLight(0x303050, 0.4);
    this.scene.add(ambient);

    // Window resize
    window.addEventListener('resize', function () {
      self.camera.aspect = window.innerWidth / window.innerHeight;
      self.camera.updateProjectionMatrix();
      self.renderer.setSize(window.innerWidth, window.innerHeight);
      if (self.composer) {
        self.composer.setSize(window.innerWidth, window.innerHeight);
      }
    });
  };

  TransformerParticleAtlas.prototype.initStarField = function () {
    var starCount = 2000;
    var starGeo = new THREE.BufferGeometry();
    var starPositions = new Float32Array(starCount * 3);
    var starSizes = new Float32Array(starCount);

    for (var i = 0; i < starCount; i++) {
      var radius = 40 + Math.random() * 60;
      var theta = Math.random() * Math.PI * 2;
      var phi = Math.acos(2 * Math.random() - 1);
      starPositions[i * 3] = radius * Math.sin(phi) * Math.cos(theta);
      starPositions[i * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      starPositions[i * 3 + 2] = radius * Math.cos(phi);
      starSizes[i] = 0.5 + Math.random() * 1.5;
    }

    starGeo.setAttribute('position', new THREE.BufferAttribute(starPositions, 3));
    starGeo.setAttribute('size', new THREE.BufferAttribute(starSizes, 1));

    var starMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 }
      },
      vertexShader: [
        'attribute float size;',
        'uniform float uTime;',
        'varying float vBrightness;',
        'void main() {',
        '  vBrightness = 0.3 + 0.7 * (0.5 + 0.5 * sin(uTime * 0.2 + position.x * 0.1));',
        '  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);',
        '  gl_PointSize = size * (80.0 / -mvPosition.z);',
        '  gl_Position = projectionMatrix * mvPosition;',
        '}'
      ].join('\n'),
      fragmentShader: [
        'varying float vBrightness;',
        'void main() {',
        '  float dist = length(gl_PointCoord - vec2(0.5));',
        '  if (dist > 0.5) discard;',
        '  float alpha = smoothstep(0.5, 0.0, dist) * vBrightness * 0.5;',
        '  gl_FragColor = vec4(0.7, 0.75, 0.9, alpha);',
        '}'
      ].join('\n'),
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    this.starField = new THREE.Points(starGeo, starMat);
    this.scene.add(this.starField);
  };

  // ─────────────────────────────────────────────
  // Layer planes
  // ─────────────────────────────────────────────
  TransformerParticleAtlas.prototype.initLayers = function () {
    var self = this;

    for (var i = 0; i < 32; i++) {
      var layerData = this.layers[i];
      var y = layerData.depth * SCENE_HEIGHT;

      // Determine zone color
      var zoneColor = COLOR_LAYER_DEFAULT.clone();
      var isZone = false;
      for (var z = 0; z < this.zones.length; z++) {
        if (i >= this.zones[z].start && i <= this.zones[z].end) {
          zoneColor = new THREE.Color(this.zones[z].color);
          isZone = true;
          break;
        }
      }

      // Layer disc
      var discGeo = new THREE.CircleGeometry(PLANE_RADIUS, 48);
      var discMat = new THREE.MeshBasicMaterial({
        color: zoneColor,
        transparent: true,
        opacity: isZone ? 0.06 : 0.015,
        side: THREE.DoubleSide,
        depthWrite: false
      });
      var disc = new THREE.Mesh(discGeo, discMat);
      disc.rotation.x = -Math.PI / 2;
      disc.position.y = y;
      disc.userData = { layerIndex: i };
      this.scene.add(disc);
      this.layerPlanes.push(disc);

      // Edge ring
      var ringGeo = new THREE.RingGeometry(PLANE_RADIUS - 0.02, PLANE_RADIUS, 64);
      var ringMat = new THREE.MeshBasicMaterial({
        color: zoneColor,
        transparent: true,
        opacity: isZone ? 0.2 : 0.04,
        side: THREE.DoubleSide,
        depthWrite: false
      });
      var ring = new THREE.Mesh(ringGeo, ringMat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = y;
      this.scene.add(ring);

      // Special: L27 pulsing halo ring (added as separate mesh)
      if (i === 27) {
        this.l27Halo = this.createL27Halo(y);
      }

      // Layer label as sprite
      var label = this.createTextSprite('L' + i, isZone ? zoneColor : new THREE.Color(0x505068));
      label.position.set(PLANE_RADIUS + 0.6, y, 0);
      label.scale.set(1.2, 0.4, 1);
      this.scene.add(label);
      this.layerLabels.push(label);
    }
  };

  TransformerParticleAtlas.prototype.createL27Halo = function (y) {
    // Outer halo ring for L27 readout
    var haloGeo = new THREE.RingGeometry(2.5, 3.2, 64);
    var haloMat = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uIntensity: { value: 0.0 } // 0 in baseline, 1 in recursive
      },
      vertexShader: [
        'varying vec2 vUv;',
        'void main() {',
        '  vUv = uv;',
        '  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
        '}'
      ].join('\n'),
      fragmentShader: [
        'uniform float uTime;',
        'uniform float uIntensity;',
        'varying vec2 vUv;',
        'void main() {',
        '  float pulse = 0.5 + 0.5 * sin(uTime * 3.0);',
        '  float ring = smoothstep(0.0, 0.3, vUv.y) * smoothstep(1.0, 0.7, vUv.y);',
        '  vec3 color = mix(vec3(0.22, 0.74, 0.97), vec3(1.0, 0.85, 0.5), uIntensity * pulse);',
        '  float alpha = ring * (0.1 + 0.3 * pulse) * max(0.15, uIntensity);',
        '  gl_FragColor = vec4(color, alpha);',
        '}'
      ].join('\n'),
      transparent: true,
      side: THREE.DoubleSide,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });
    var halo = new THREE.Mesh(haloGeo, haloMat);
    halo.rotation.x = -Math.PI / 2;
    halo.position.y = y;
    this.scene.add(halo);
    return halo;
  };

  TransformerParticleAtlas.prototype.createTextSprite = function (text, color) {
    var canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 48;
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, 128, 48);
    ctx.font = 'bold 24px Inter, sans-serif';
    ctx.fillStyle = '#' + color.getHexString();
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, 4, 24);

    var tex = new THREE.CanvasTexture(canvas);
    tex.minFilter = THREE.LinearFilter;
    var mat = new THREE.SpriteMaterial({
      map: tex,
      transparent: true,
      depthWrite: false
    });
    return new THREE.Sprite(mat);
  };

  // ─────────────────────────────────────────────
  // Compute particle positions for both modes
  // ─────────────────────────────────────────────
  TransformerParticleAtlas.prototype.computeParticlePositions = function () {
    var totalCount = TOTAL_PARTICLES;
    this.baselinePositions = new Float32Array(totalCount * 3);
    this.recursivePositions = new Float32Array(totalCount * 3);
    this.baselineColors = new Float32Array(totalCount * 3);
    this.recursiveColors = new Float32Array(totalCount * 3);
    this.baselineSizes = new Float32Array(totalCount);
    this.recursiveSizes = new Float32Array(totalCount);

    var baseTrajectory = this.controlModes.baseline.trajectory;
    var recTrajectory = this.controlModes.recursive.trajectory;

    // Precompute noise seeds for consistent particle identity
    this.noiseSeedsX = new Float32Array(totalCount);
    this.noiseSeedsZ = new Float32Array(totalCount);
    this.noisePhase = new Float32Array(totalCount);
    for (var s = 0; s < totalCount; s++) {
      this.noiseSeedsX[s] = (Math.random() - 0.5) * 2;
      this.noiseSeedsZ[s] = (Math.random() - 0.5) * 2;
      this.noisePhase[s] = Math.random() * Math.PI * 2;
    }

    for (var layer = 0; layer < 32; layer++) {
      var layerData = this.layers[layer];
      var baseTraj = baseTrajectory[layer];
      var recTraj = recTrajectory[layer];

      // Field strength determines baseline spread (inverse: low field = wide)
      var fieldStr = layerData.field_strength;
      // v_proj_d magnitude determines particle size (inverse: large |v_proj_d| = small = contracted)
      var vprojMag = Math.abs(layerData.v_proj_d);

      // Baseline spread: inversely proportional to field_strength
      // Range from ~1.5 (high field) to ~3.0 (low field)
      var baseSpread = 0.8 + 2.2 * (1.0 - Math.min(fieldStr, 1.0));

      // Self-referential spread depends on layer position
      var recSpread;
      if (layer <= 5) {
        // Early layers: WIDE — active processing zone
        recSpread = 1.5 + 1.5 * fieldStr;
      } else if (layer <= 24) {
        // Mid layers: progressively converging
        var t = (layer - 6) / 18;
        recSpread = 2.5 * (1.0 - t * 0.7);
      } else if (layer === 25) {
        // Controller: noticeable tightening
        recSpread = 0.6;
      } else if (layer === 27) {
        // READOUT: COLLAPSE into tight beam — THE R_V CONTRACTION
        recSpread = 0.12;
      } else if (layer === 26) {
        recSpread = 0.4;
      } else {
        // L28-31: gradually relax
        recSpread = 0.3 + (layer - 28) * 0.15;
      }

      // Particle size based on v_proj_d magnitude
      // Base size, inverted for contraction zones
      var baseSize = 3.0 + 2.0 * (1.0 - Math.min(vprojMag / 2.5, 1.0));
      var recSize;
      if (layer === 27) {
        recSize = 1.5; // tight, small, concentrated
      } else if (vprojMag > 1.0) {
        recSize = 2.0;
      } else {
        recSize = 3.5 + 1.5 * (1.0 - Math.min(fieldStr, 1.0));
      }

      // Colors
      var baseColor = COLOR_BASELINE.clone();
      // Modulate baseline slightly by field_strength
      var fieldDim = 0.6 + 0.4 * fieldStr;
      baseColor.multiplyScalar(fieldDim);

      var recColor;
      if (layer === 27) {
        recColor = COLOR_RECURSIVE_HOT.clone();
      } else if (layer >= 25 && layer <= 27) {
        recColor = COLOR_RECURSIVE_COOL.clone().lerp(COLOR_RECURSIVE_HOT, 0.5);
      } else {
        recColor = COLOR_RECURSIVE_COOL.clone();
        var warmth = Math.min(fieldStr, 1.0);
        recColor.lerp(COLOR_RECURSIVE_HOT, warmth * 0.3);
      }

      var startIdx = layer * PARTICLES_PER_LAYER;
      for (var p = 0; p < PARTICLES_PER_LAYER; p++) {
        var idx = startIdx + p;
        var i3 = idx * 3;

        // Gaussian-ish noise (Box-Muller approximation)
        var nX = this.noiseSeedsX[idx];
        var nZ = this.noiseSeedsZ[idx];
        // Scale noise radially for more natural distribution
        var r = Math.sqrt(nX * nX + nZ * nZ);
        var gauss = r > 0 ? Math.exp(-r * r * 0.5) * 1.5 : 1.0;
        var noiseX = nX * gauss;
        var noiseZ = nZ * gauss;

        var y = layerData.depth * SCENE_HEIGHT;

        // Baseline positions
        this.baselinePositions[i3] = baseTraj.x * POSITION_SCALE + noiseX * baseSpread;
        this.baselinePositions[i3 + 1] = y + (Math.random() - 0.5) * LAYER_SPACING * 0.3;
        this.baselinePositions[i3 + 2] = baseTraj.z * POSITION_SCALE + noiseZ * baseSpread;

        // Recursive positions
        this.recursivePositions[i3] = recTraj.x * POSITION_SCALE + noiseX * recSpread;
        this.recursivePositions[i3 + 1] = y + (Math.random() - 0.5) * LAYER_SPACING * 0.3;
        this.recursivePositions[i3 + 2] = recTraj.z * POSITION_SCALE + noiseZ * recSpread;

        // Colors
        // Add per-particle color variation
        var colorJitter = 0.9 + Math.random() * 0.2;
        this.baselineColors[i3] = baseColor.r * colorJitter;
        this.baselineColors[i3 + 1] = baseColor.g * colorJitter;
        this.baselineColors[i3 + 2] = baseColor.b * colorJitter;

        this.recursiveColors[i3] = recColor.r * colorJitter;
        this.recursiveColors[i3 + 1] = recColor.g * colorJitter;
        this.recursiveColors[i3 + 2] = recColor.b * colorJitter;

        // Sizes
        var sizeJitter = 0.7 + Math.random() * 0.6;
        this.baselineSizes[idx] = baseSize * sizeJitter;
        this.recursiveSizes[idx] = recSize * sizeJitter;
      }
    }
  };

  // ─────────────────────────────────────────────
  // Particle system
  // ─────────────────────────────────────────────
  TransformerParticleAtlas.prototype.initParticles = function () {
    var geometry = new THREE.BufferGeometry();
    var totalCount = TOTAL_PARTICLES;

    // Start with baseline positions
    var positions = new Float32Array(totalCount * 3);
    var colors = new Float32Array(totalCount * 3);
    var sizes = new Float32Array(totalCount);

    for (var i = 0; i < totalCount * 3; i++) {
      positions[i] = this.baselinePositions[i];
      colors[i] = this.baselineColors[i];
    }
    for (var j = 0; j < totalCount; j++) {
      sizes[j] = this.baselineSizes[j];
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));

    // Custom shader material for particles
    var material = new THREE.ShaderMaterial({
      uniforms: {
        uTime: { value: 0 },
        uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
        uGlobalAlpha: { value: 1.0 }
      },
      vertexShader: [
        'attribute float size;',
        'attribute vec3 color;',
        'uniform float uTime;',
        'uniform float uPixelRatio;',
        'varying vec3 vColor;',
        'varying float vAlpha;',
        'void main() {',
        '  vColor = color;',
        '  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);',
        '  float dist = -mvPosition.z;',
        '  gl_PointSize = size * uPixelRatio * (120.0 / dist);',
        '  gl_PointSize = clamp(gl_PointSize, 1.0, 40.0);',
        // Subtle breathing based on time
        '  gl_PointSize *= 0.95 + 0.05 * sin(uTime * 1.5 + position.y * 2.0 + position.x);',
        '  vAlpha = smoothstep(80.0, 5.0, dist);',
        '  gl_Position = projectionMatrix * mvPosition;',
        '}'
      ].join('\n'),
      fragmentShader: [
        'varying vec3 vColor;',
        'varying float vAlpha;',
        'uniform float uGlobalAlpha;',
        'void main() {',
        '  float dist = length(gl_PointCoord - vec2(0.5));',
        '  if (dist > 0.5) discard;',
        // Soft glow falloff
        '  float core = smoothstep(0.5, 0.05, dist);',
        '  float glow = smoothstep(0.5, 0.2, dist) * 0.6;',
        '  float brightness = core + glow;',
        '  vec3 finalColor = vColor * brightness;',
        '  float alpha = brightness * vAlpha * uGlobalAlpha;',
        '  gl_FragColor = vec4(finalColor, alpha);',
        '}'
      ].join('\n'),
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    this.particleSystem = new THREE.Points(geometry, material);
    this.scene.add(this.particleSystem);
    this.particleGeometry = geometry;
  };

  // ─────────────────────────────────────────────
  // Trajectory lines
  // ─────────────────────────────────────────────
  TransformerParticleAtlas.prototype.initTrajectories = function () {
    this.trajectoryGroup = new THREE.Group();
    this.scene.add(this.trajectoryGroup);

    // Baseline trajectory
    this.baselineLine = this.createTrajectoryLine(
      this.controlModes.baseline.trajectory,
      0x4488ff, 0.4
    );
    this.trajectoryGroup.add(this.baselineLine);

    // Recursive trajectory
    this.recursiveLine = this.createTrajectoryLine(
      this.controlModes.recursive.trajectory,
      0xffaa22, 0.4
    );
    this.trajectoryGroup.add(this.recursiveLine);

    // Connection lines between corresponding points (showing divergence)
    this.divergenceLines = this.createDivergenceLines();
    this.trajectoryGroup.add(this.divergenceLines);
  };

  TransformerParticleAtlas.prototype.createTrajectoryLine = function (trajectory, color, opacity) {
    var points = [];
    for (var i = 0; i < trajectory.length; i++) {
      var t = trajectory[i];
      points.push(new THREE.Vector3(
        t.x * POSITION_SCALE,
        t.y * SCENE_HEIGHT,
        t.z * POSITION_SCALE
      ));
    }

    // Smooth curve through points
    var curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.3);
    var curvePoints = curve.getPoints(128);
    var geometry = new THREE.BufferGeometry().setFromPoints(curvePoints);

    // Per-vertex alpha: brighter near zone layers
    var alphas = new Float32Array(curvePoints.length);
    for (var j = 0; j < curvePoints.length; j++) {
      var t2 = j / (curvePoints.length - 1);
      var layerApprox = t2 * 31;
      var isNearZone = false;
      for (var z = 0; z < this.zones.length; z++) {
        if (Math.abs(layerApprox - this.zones[z].start) < 2 ||
            Math.abs(layerApprox - this.zones[z].end) < 2) {
          isNearZone = true;
          break;
        }
      }
      alphas[j] = isNearZone ? 1.0 : 0.5;
    }
    geometry.setAttribute('alpha', new THREE.BufferAttribute(alphas, 1));

    var material = new THREE.ShaderMaterial({
      uniforms: {
        uColor: { value: new THREE.Color(color) },
        uOpacity: { value: opacity },
        uTime: { value: 0 }
      },
      vertexShader: [
        'attribute float alpha;',
        'varying float vAlpha;',
        'void main() {',
        '  vAlpha = alpha;',
        '  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);',
        '}'
      ].join('\n'),
      fragmentShader: [
        'uniform vec3 uColor;',
        'uniform float uOpacity;',
        'varying float vAlpha;',
        'void main() {',
        '  gl_FragColor = vec4(uColor, uOpacity * vAlpha);',
        '}'
      ].join('\n'),
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    return new THREE.Line(geometry, material);
  };

  TransformerParticleAtlas.prototype.createDivergenceLines = function () {
    var baseTraj = this.controlModes.baseline.trajectory;
    var recTraj = this.controlModes.recursive.trajectory;
    var positions = [];
    var colors = [];

    for (var i = 0; i < 32; i++) {
      var bt = baseTraj[i];
      var rt = recTraj[i];

      positions.push(
        bt.x * POSITION_SCALE, bt.y * SCENE_HEIGHT, bt.z * POSITION_SCALE,
        rt.x * POSITION_SCALE, rt.y * SCENE_HEIGHT, rt.z * POSITION_SCALE
      );

      // Color gradient: cool to warm showing divergence distance
      var dx = bt.x - rt.x;
      var dz = bt.z - rt.z;
      var dist = Math.sqrt(dx * dx + dz * dz);
      var warmth = Math.min(dist / 0.5, 1.0);

      var c1 = new THREE.Color(0x4488ff).lerp(new THREE.Color(0xffaa22), warmth * 0.3);
      var c2 = new THREE.Color(0xffaa22).lerp(new THREE.Color(0xff6644), warmth * 0.5);

      colors.push(c1.r, c1.g, c1.b, c2.r, c2.g, c2.b);
    }

    var geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    var mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.12,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    return new THREE.LineSegments(geo, mat);
  };

  // ─────────────────────────────────────────────
  // Trail system (fading echoes)
  // ─────────────────────────────────────────────
  TransformerParticleAtlas.prototype.initTrailSystem = function () {
    // We'll store previous position snapshots and render as faded point clouds
    for (var t = 0; t < TRAIL_LENGTH; t++) {
      var geo = new THREE.BufferGeometry();
      var pos = new Float32Array(TOTAL_PARTICLES * 3);
      var sizes = new Float32Array(TOTAL_PARTICLES);
      var col = new Float32Array(TOTAL_PARTICLES * 3);

      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));

      var fadeAlpha = 0.15 * (1.0 - t / TRAIL_LENGTH);

      var mat = new THREE.ShaderMaterial({
        uniforms: {
          uPixelRatio: { value: Math.min(window.devicePixelRatio, 2) },
          uAlpha: { value: fadeAlpha }
        },
        vertexShader: [
          'attribute float size;',
          'attribute vec3 color;',
          'uniform float uPixelRatio;',
          'varying vec3 vColor;',
          'void main() {',
          '  vColor = color;',
          '  vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);',
          '  gl_PointSize = size * uPixelRatio * 0.5 * (120.0 / -mvPosition.z);',
          '  gl_PointSize = clamp(gl_PointSize, 0.5, 20.0);',
          '  gl_Position = projectionMatrix * mvPosition;',
          '}'
        ].join('\n'),
        fragmentShader: [
          'varying vec3 vColor;',
          'uniform float uAlpha;',
          'void main() {',
          '  float dist = length(gl_PointCoord - vec2(0.5));',
          '  if (dist > 0.5) discard;',
          '  float a = smoothstep(0.5, 0.1, dist) * uAlpha;',
          '  gl_FragColor = vec4(vColor, a);',
          '}'
        ].join('\n'),
        transparent: true,
        depthWrite: false,
        blending: THREE.AdditiveBlending
      });

      var trailPoints = new THREE.Points(geo, mat);
      trailPoints.visible = false;
      this.scene.add(trailPoints);
      this.trailMeshes.push(trailPoints);
      this.trailHistory.push(null); // will be filled during animate
    }
  };

  // ─────────────────────────────────────────────
  // Post-processing (Bloom)
  // ─────────────────────────────────────────────
  TransformerParticleAtlas.prototype.initPostProcessing = function () {
    if (!THREE.EffectComposer || !THREE.RenderPass || !THREE.UnrealBloomPass || !THREE.ShaderPass) {
      console.warn('Post-processing classes not loaded, using standard rendering.');
      this.composer = null;
      this.bloomPass = null;
      return;
    }
    try {
      this.composer = new THREE.EffectComposer(this.renderer);
      var renderPass = new THREE.RenderPass(this.scene, this.camera);
      this.composer.addPass(renderPass);

      this.bloomPass = new THREE.UnrealBloomPass(
        new THREE.Vector2(window.innerWidth, window.innerHeight),
        0.6,    // strength
        0.4,    // radius
        0.85    // threshold
      );
      this.composer.addPass(this.bloomPass);
    } catch (e) {
      console.warn('Post-processing unavailable, falling back to standard rendering:', e);
      this.composer = null;
      this.bloomPass = null;
    }
  };

  // ─────────────────────────────────────────────
  // HUD wiring
  // ─────────────────────────────────────────────
  TransformerParticleAtlas.prototype.initHUD = function () {
    this.updateMetricsPanel('baseline');
    this.updateExemplarPanel('baseline');
  };

  TransformerParticleAtlas.prototype.updateMetricsPanel = function (mode) {
    var metrics = this.controlModes[mode === 'recursive' ? 'recursive' : 'baseline'].metrics;

    document.getElementById('rv-value').textContent = metrics.mean_output_rv.toFixed(3);
    document.getElementById('effect-value').textContent =
      this.layers[27].v_proj_d.toFixed(2);
    document.getElementById('n-recursive').textContent =
      this.controlModes.recursive.metrics.n;
    document.getElementById('n-baseline').textContent =
      this.controlModes.baseline.metrics.n;
    document.getElementById('bt-art-value').textContent =
      (metrics.bt_art_rate * 100).toFixed(1) + '%';

    // Class counts
    var countsEl = document.getElementById('class-counts');
    var counts = metrics.class_counts;
    var html = '';
    var classColors = {
      'BREAKTHROUGH': '#34d399',
      'ARTICULATE': '#93c5fd',
      'CONCEPTUAL': '#c084fc',
      'SURFACE': '#94a3b8',
      'REPETITIVE': '#fb923c'
    };
    for (var cls in counts) {
      if (counts.hasOwnProperty(cls) && counts[cls] > 0) {
        html += '<div class="metric-row">' +
          '<span class="metric-label" style="color:' + (classColors[cls] || '#707088') + '">' +
          cls + '</span>' +
          '<span class="metric-value count">' + counts[cls] + '</span></div>';
      }
    }
    countsEl.innerHTML = html;
  };

  TransformerParticleAtlas.prototype.updateExemplarPanel = function (mode) {
    var exemplar = this.controlModes[mode === 'recursive' ? 'recursive' : 'baseline'].exemplar;

    var classEl = document.getElementById('exemplar-classification');
    classEl.textContent = exemplar.classification;
    classEl.className = exemplar.classification;

    document.getElementById('exemplar-prompt').textContent =
      '"' + this.truncateText(exemplar.prompt_text, 200) + '"';
    document.getElementById('exemplar-output').textContent =
      this.truncateText(exemplar.generated_text, 300);
  };

  TransformerParticleAtlas.prototype.truncateText = function (text, maxLen) {
    if (text.length <= maxLen) return text;
    return text.substring(0, maxLen) + '...';
  };

  // ─────────────────────────────────────────────
  // Interaction
  // ─────────────────────────────────────────────
  TransformerParticleAtlas.prototype.initInteraction = function () {
    var self = this;

    // Mode buttons
    var buttons = document.querySelectorAll('.mode-btn');
    buttons.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var mode = this.getAttribute('data-mode');
        self.setMode(mode);
      });
    });

    // Trajectory toggle
    document.getElementById('show-trajectories').addEventListener('change', function () {
      self.showTrajectories = this.checked;
      self.trajectoryGroup.visible = this.checked;
    });

    // Trail toggle
    document.getElementById('show-trails').addEventListener('change', function () {
      self.showTrails = this.checked;
      for (var i = 0; i < self.trailMeshes.length; i++) {
        self.trailMeshes[i].visible = this.checked;
      }
    });

    // Auto-rotate toggle
    document.getElementById('auto-rotate').addEventListener('change', function () {
      self.autoRotateEnabled = this.checked;
    });

    // Mouse move for layer hover
    window.addEventListener('mousemove', function (e) {
      self.mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
      self.mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;
      self.mouseScreen = { x: e.clientX, y: e.clientY };
      self.idleTime = 0;
      self.isUserInteracting = true;
    });

    // Click for layer focus
    window.addEventListener('click', function (e) {
      // Ignore clicks on HUD elements
      if (e.target.closest('#hud > *')) return;
      self.handleLayerClick();
    });

    // ESC to reset focus
    window.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        self.resetFocus();
      }
    });

    // Orbit controls interaction tracking
    this.controls.addEventListener('start', function () {
      self.isUserInteracting = true;
      self.idleTime = 0;
    });

    this.controls.addEventListener('end', function () {
      // Will resume auto-rotate after idle timeout
    });
  };

  TransformerParticleAtlas.prototype.handleLayerClick = function () {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    var intersects = this.raycaster.intersectObjects(this.layerPlanes);

    if (intersects.length > 0) {
      var layerIdx = intersects[0].object.userData.layerIndex;
      this.focusOnLayer(layerIdx);
    }
  };

  TransformerParticleAtlas.prototype.focusOnLayer = function (layerIdx) {
    this.focusedLayer = layerIdx;
    var layerData = this.layers[layerIdx];
    var y = layerData.depth * SCENE_HEIGHT;

    // Determine a nice camera position for this layer
    var targetPos = new THREE.Vector3(
      this.controls.target.x,
      y,
      this.controls.target.z
    );

    // Animate camera to focus on this layer
    var camTarget = new THREE.Vector3(6, y + 1.5, 8);
    this.animateCamera(camTarget, targetPos, 1000);

    // Show focus hint
    var hint = document.getElementById('focus-hint');
    hint.classList.add('visible');
    setTimeout(function () { hint.classList.remove('visible'); }, 2000);
  };

  TransformerParticleAtlas.prototype.resetFocus = function () {
    this.focusedLayer = -1;
    var defaultTarget = new THREE.Vector3(0, SCENE_HEIGHT * 0.45, 0);
    var defaultCam = new THREE.Vector3(12, 10, 16);
    this.animateCamera(defaultCam, defaultTarget, 1000);
  };

  TransformerParticleAtlas.prototype.animateCamera = function (targetPos, lookAt, duration) {
    var self = this;
    var startPos = self.camera.position.clone();
    var startTarget = self.controls.target.clone();
    var startTime = performance.now();

    function step(now) {
      var t = Math.min((now - startTime) / duration, 1.0);
      // Ease in-out cubic
      t = t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

      self.camera.position.lerpVectors(startPos, targetPos, t);
      self.controls.target.lerpVectors(startTarget, lookAt, t);

      if (t < 1.0) {
        requestAnimationFrame(step);
      }
    }
    requestAnimationFrame(step);
  };

  // ─────────────────────────────────────────────
  // Mode switching
  // ─────────────────────────────────────────────
  TransformerParticleAtlas.prototype.setMode = function (mode) {
    // Update button states
    var buttons = document.querySelectorAll('.mode-btn');
    buttons.forEach(function (btn) {
      btn.className = 'mode-btn';
      if (btn.getAttribute('data-mode') === mode) {
        btn.classList.add('active-' + mode);
      }
    });

    // Update mode indicator
    var indicator = document.getElementById('mode-indicator');
    indicator.className = mode;
    var modeLabels = {
      'baseline': 'Baseline Mode',
      'recursive': 'Self-Referential Mode',
      'transition': 'Transition (Morphing...)'
    };
    indicator.textContent = modeLabels[mode] || mode;

    if (mode === 'transition') {
      // Start continuous ping-pong transition
      this.isTransitioning = true;
      this.transitionProgress = 0;
      this.transitionDirection = 1;
      this.currentMode = 'baseline';
      this.targetMode = 'recursive';
      // Update HUD to show recursive metrics (the dramatic state)
      this.updateMetricsPanel('recursive');
      this.updateExemplarPanel('recursive');
    } else if (mode === 'baseline') {
      if (this.currentMode !== 'baseline' || this.isTransitioning) {
        this.isTransitioning = true;
        this.transitionProgress = 0;
        this.transitionDirection = 1;
        this.currentMode = this.currentMode;
        this.targetMode = 'baseline';
      }
      this.updateMetricsPanel('baseline');
      this.updateExemplarPanel('baseline');
    } else if (mode === 'recursive') {
      if (this.currentMode !== 'recursive' || this.isTransitioning) {
        this.isTransitioning = true;
        this.transitionProgress = 0;
        this.transitionDirection = 1;
        this.currentMode = this.currentMode;
        this.targetMode = 'recursive';
      }
      this.updateMetricsPanel('recursive');
      this.updateExemplarPanel('recursive');
    }

    this.pendingMode = mode;
  };

  // ─────────────────────────────────────────────
  // Animation loop
  // ─────────────────────────────────────────────
  TransformerParticleAtlas.prototype.animate = function () {
    var self = this;
    var frameCount = 0;
    var trailTimer = 0;

    function loop() {
      requestAnimationFrame(loop);

      // Skip if context is lost
      if (self._contextLost) return;

      var dt = Math.min(self.clock.getDelta(), 0.1); // Cap dt to prevent huge jumps after tab suspend
      var elapsed = self.clock.elapsedTime;
      frameCount++;

      try {
      // Idle tracking
      self.idleTime += dt;
      if (self.idleTime > 3.0) {
        self.isUserInteracting = false;
      }

      // Auto-rotate
      self.controls.autoRotate = self.autoRotateEnabled && !self.isUserInteracting;
      self.controls.update();

      // Update transition
      self.updateTransition(dt);

      // Update particles (brownian motion)
      self.updateParticles(dt, elapsed);

      // Update trail snapshots every ~4 frames
      trailTimer += dt;
      if (self.showTrails && trailTimer > 0.08) {
        self.updateTrails();
        trailTimer = 0;
      }

      // Update star field time
      if (self.starField && self.starField.material.uniforms) {
        self.starField.material.uniforms.uTime.value = elapsed;
      }

      // Update L27 halo
      if (self.l27Halo && self.l27Halo.material.uniforms) {
        self.l27Halo.material.uniforms.uTime.value = elapsed;
        var recIntensity = self.getRecursiveBlend();
        self.l27Halo.material.uniforms.uIntensity.value = recIntensity;
      }

      // Update particle shader time
      if (self.particleSystem && self.particleSystem.material.uniforms) {
        self.particleSystem.material.uniforms.uTime.value = elapsed;
      }

      // Trajectory line time
      if (self.baselineLine && self.baselineLine.material.uniforms) {
        self.baselineLine.material.uniforms.uTime.value = elapsed;
      }
      if (self.recursiveLine && self.recursiveLine.material.uniforms) {
        self.recursiveLine.material.uniforms.uTime.value = elapsed;
      }

      // Trajectory visibility based on current mode
      if (self.baselineLine && self.recursiveLine) {
        var blend = self.getRecursiveBlend();
        self.baselineLine.material.uniforms.uOpacity.value = 0.4 * (1.0 - blend * 0.5);
        self.recursiveLine.material.uniforms.uOpacity.value = 0.15 + 0.45 * blend;
        if (self.divergenceLines) {
          self.divergenceLines.material.opacity = 0.06 + 0.15 * blend;
        }
      }

      // Update bloom strength based on mode
      if (self.bloomPass) {
        var recBlend = self.getRecursiveBlend();
        self.bloomPass.strength = 0.4 + 0.5 * recBlend;
      }

      // Layer hover raycasting (every other frame)
      if (frameCount % 2 === 0) {
        self.updateLayerHover();
      }

      // Render
      if (self.composer) {
        self.composer.render();
      } else {
        self.renderer.render(self.scene, self.camera);
      }

      } catch (e) {
        // Log but don't stop the loop — one bad frame shouldn't kill everything
        if (frameCount % 60 === 0) console.error('Render error:', e);
      }
    }

    loop();
  };

  TransformerParticleAtlas.prototype.getRecursiveBlend = function () {
    // Returns 0.0 (fully baseline) to 1.0 (fully recursive)
    if (!this.isTransitioning && this.currentMode === 'baseline') return 0.0;
    if (!this.isTransitioning && this.currentMode === 'recursive') return 1.0;

    // During transition, depends on direction and target
    if (this.targetMode === 'recursive') {
      return this.transitionProgress;
    } else if (this.targetMode === 'baseline') {
      return 1.0 - this.transitionProgress;
    }
    // Ping-pong: transitionDirection tells us
    if (this.transitionDirection > 0) {
      return this.transitionProgress;
    } else {
      return 1.0 - this.transitionProgress;
    }
  };

  TransformerParticleAtlas.prototype.updateTransition = function (dt) {
    if (!this.isTransitioning) return;

    this.transitionProgress += dt / TRANSITION_DURATION;

    if (this.transitionProgress >= 1.0) {
      this.transitionProgress = 1.0;

      if (this.pendingMode === 'transition') {
        // Ping-pong: reverse direction
        this.transitionDirection *= -1;
        this.transitionProgress = 0;
        var temp = this.currentMode;
        this.currentMode = this.targetMode;
        this.targetMode = temp;
      } else {
        // Single transition complete
        this.isTransitioning = false;
        this.currentMode = this.targetMode;
      }
    }
  };

  TransformerParticleAtlas.prototype.updateParticles = function (dt, elapsed) {
    var posAttr = this.particleGeometry.getAttribute('position');
    var colAttr = this.particleGeometry.getAttribute('color');
    var sizeAttr = this.particleGeometry.getAttribute('size');
    var positions = posAttr.array;
    var colors = colAttr.array;
    var sizes = sizeAttr.array;

    // Determine blend factor (0 = fully current source, 1 = fully target)
    var t = this.isTransitioning ? this.transitionProgress : 1.0;
    // Ease: smooth step for more dramatic effect
    var eased = t * t * (3.0 - 2.0 * t);

    // Determine source and target arrays
    var srcPos, tgtPos, srcCol, tgtCol, srcSize, tgtSize;

    if (this.isTransitioning) {
      if (this.targetMode === 'recursive') {
        srcPos = this.baselinePositions;
        tgtPos = this.recursivePositions;
        srcCol = this.baselineColors;
        tgtCol = this.recursiveColors;
        srcSize = this.baselineSizes;
        tgtSize = this.recursiveSizes;
      } else {
        srcPos = this.recursivePositions;
        tgtPos = this.baselinePositions;
        srcCol = this.recursiveColors;
        tgtCol = this.baselineColors;
        srcSize = this.recursiveSizes;
        tgtSize = this.baselineSizes;
      }
    } else {
      if (this.currentMode === 'recursive') {
        srcPos = this.recursivePositions;
        tgtPos = this.recursivePositions;
        srcCol = this.recursiveColors;
        tgtCol = this.recursiveColors;
        srcSize = this.recursiveSizes;
        tgtSize = this.recursiveSizes;
      } else {
        srcPos = this.baselinePositions;
        tgtPos = this.baselinePositions;
        srcCol = this.baselineColors;
        tgtCol = this.baselineColors;
        srcSize = this.baselineSizes;
        tgtSize = this.baselineSizes;
      }
    }

    // Update each particle
    for (var i = 0; i < TOTAL_PARTICLES; i++) {
      var i3 = i * 3;

      // Interpolate position
      var targetX = srcPos[i3] + (tgtPos[i3] - srcPos[i3]) * eased;
      var targetY = srcPos[i3 + 1] + (tgtPos[i3 + 1] - srcPos[i3 + 1]) * eased;
      var targetZ = srcPos[i3 + 2] + (tgtPos[i3 + 2] - srcPos[i3 + 2]) * eased;

      // Add brownian motion
      var phase = this.noisePhase[i] + elapsed * DRIFT_SPEED;
      var bx = Math.sin(phase * 1.3 + i * 0.01) * BROWNIAN_STRENGTH;
      var by = Math.cos(phase * 0.7 + i * 0.02) * BROWNIAN_STRENGTH * 0.3;
      var bz = Math.sin(phase * 1.1 + i * 0.015) * BROWNIAN_STRENGTH;

      positions[i3] = targetX + bx;
      positions[i3 + 1] = targetY + by;
      positions[i3 + 2] = targetZ + bz;

      // Interpolate color
      colors[i3] = srcCol[i3] + (tgtCol[i3] - srcCol[i3]) * eased;
      colors[i3 + 1] = srcCol[i3 + 1] + (tgtCol[i3 + 1] - srcCol[i3 + 1]) * eased;
      colors[i3 + 2] = srcCol[i3 + 2] + (tgtCol[i3 + 2] - srcCol[i3 + 2]) * eased;

      // Interpolate size
      sizes[i] = srcSize[i] + (tgtSize[i] - srcSize[i]) * eased;
    }

    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    sizeAttr.needsUpdate = true;
  };

  TransformerParticleAtlas.prototype.updateTrails = function () {
    // Rotate ring buffer index instead of shifting arrays
    if (this._trailHead === undefined) {
      this._trailHead = 0;
      // Pre-allocate all trail buffers once
      this._trailBuffers = [];
      for (var b = 0; b < TRAIL_LENGTH; b++) {
        this._trailBuffers.push({
          positions: new Float32Array(TOTAL_PARTICLES * 3),
          colors: new Float32Array(TOTAL_PARTICLES * 3),
          sizes: new Float32Array(TOTAL_PARTICLES),
          filled: false
        });
      }
    }

    // Copy current data into current head slot (reuses existing buffer — zero allocation)
    var buf = this._trailBuffers[this._trailHead];
    buf.positions.set(this.particleGeometry.getAttribute('position').array);
    buf.colors.set(this.particleGeometry.getAttribute('color').array);
    buf.sizes.set(this.particleGeometry.getAttribute('size').array);
    buf.filled = true;

    // Update trail meshes from ring buffer (newest to oldest)
    for (var i = 0; i < TRAIL_LENGTH; i++) {
      var bufIdx = (this._trailHead - i + TRAIL_LENGTH) % TRAIL_LENGTH;
      var trail = this.trailMeshes[i];
      var data = this._trailBuffers[bufIdx];

      if (!data.filled) {
        trail.visible = false;
        continue;
      }
      trail.visible = this.showTrails;

      var geo = trail.geometry;
      geo.getAttribute('position').array.set(data.positions);
      geo.getAttribute('position').needsUpdate = true;
      geo.getAttribute('color').array.set(data.colors);
      geo.getAttribute('color').needsUpdate = true;
      geo.getAttribute('size').array.set(data.sizes);
      geo.getAttribute('size').needsUpdate = true;
    }

    this._trailHead = (this._trailHead + 1) % TRAIL_LENGTH;
  };

  TransformerParticleAtlas.prototype.updateLayerHover = function () {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    var intersects = this.raycaster.intersectObjects(this.layerPlanes);

    var tooltip = document.getElementById('layer-tooltip');

    if (intersects.length > 0) {
      var layerIdx = intersects[0].object.userData.layerIndex;

      if (layerIdx !== this.hoveredLayer) {
        this.hoveredLayer = layerIdx;
        var layerData = this.layers[layerIdx];

        // Determine zone name
        var zoneName = '';
        for (var z = 0; z < this.zones.length; z++) {
          if (layerIdx >= this.zones[z].start && layerIdx <= this.zones[z].end) {
            zoneName = ' (' + this.zones[z].label + ')';
            break;
          }
        }

        document.getElementById('tt-layer-name').textContent =
          'Layer ' + layerIdx + zoneName;
        document.getElementById('tt-residual').textContent =
          layerData.residual_d.toFixed(3);
        document.getElementById('tt-vproj').textContent =
          layerData.v_proj_d.toFixed(3);
        document.getElementById('tt-mlp').textContent =
          layerData.mlp_d.toFixed(3);
        document.getElementById('tt-field').textContent =
          layerData.field_strength.toFixed(4);

        // Color the v_proj value by magnitude
        var vprojEl = document.getElementById('tt-vproj');
        if (layerData.v_proj_d < -1.0) {
          vprojEl.style.color = '#f87171'; // red for strong contraction
        } else if (layerData.v_proj_d > 1.0) {
          vprojEl.style.color = '#34d399'; // green for expansion
        } else {
          vprojEl.style.color = '#c0c0d8';
        }

        // Highlight the hovered layer plane
        this.highlightLayer(layerIdx, true);
      }

      // Position tooltip near mouse
      if (this.mouseScreen) {
        tooltip.style.display = 'block';
        tooltip.style.left = (this.mouseScreen.x + 16) + 'px';
        tooltip.style.top = (this.mouseScreen.y - 20) + 'px';

        // Keep within viewport
        var rect = tooltip.getBoundingClientRect();
        if (rect.right > window.innerWidth - 10) {
          tooltip.style.left = (this.mouseScreen.x - rect.width - 16) + 'px';
        }
        if (rect.bottom > window.innerHeight - 10) {
          tooltip.style.top = (this.mouseScreen.y - rect.height - 10) + 'px';
        }
      }
    } else {
      if (this.hoveredLayer >= 0) {
        this.highlightLayer(this.hoveredLayer, false);
        this.hoveredLayer = -1;
      }
      tooltip.style.display = 'none';
    }
  };

  TransformerParticleAtlas.prototype.highlightLayer = function (layerIdx, highlight) {
    var disc = this.layerPlanes[layerIdx];
    if (!disc) return;

    var isZone = false;
    for (var z = 0; z < this.zones.length; z++) {
      if (layerIdx >= this.zones[z].start && layerIdx <= this.zones[z].end) {
        isZone = true;
        break;
      }
    }

    if (highlight) {
      disc.material.opacity = isZone ? 0.15 : 0.06;
    } else {
      disc.material.opacity = isZone ? 0.06 : 0.015;
    }
  };

  // ─────────────────────────────────────────────
  // Bootstrap
  // ─────────────────────────────────────────────
  function boot() {
    var overlay = document.getElementById('loading-overlay');

    if (typeof THREE === 'undefined') {
      console.error('Three.js not loaded. Check CDN connectivity.');
      if (overlay) overlay.innerHTML = '<h2 style="color:#f87171;">Error: Three.js failed to load (check internet)</h2>';
      return;
    }

    var data = window.MISTRAL_CONTROL_ATLAS_DATA;
    if (!data) {
      console.error('MISTRAL_CONTROL_ATLAS_DATA not found. Ensure data script is loaded.');
      if (overlay) overlay.innerHTML = '<h2 style="color:#f87171;">Error: Data file not loaded</h2>';
      return;
    }

    if (!data.anchor_bundle || !data.anchor_bundle.control || !data.anchor_bundle.control.modes) {
      console.error('Data structure missing anchor_bundle.control.modes');
      if (overlay) overlay.innerHTML = '<h2 style="color:#f87171;">Error: Data structure invalid</h2>';
      return;
    }

    var canvas = document.getElementById('atlas-canvas');
    if (!canvas) {
      console.error('Canvas element #atlas-canvas not found.');
      return;
    }

    try {
      window.atlas = new TransformerParticleAtlas(canvas, data);
    } catch (e) {
      console.error('Atlas initialization failed:', e);
      if (overlay) overlay.innerHTML = '<h2 style="color:#f87171;">Error: ' + e.message + '</h2>';
    }
  }

  // Wait for DOM and all scripts
  if (document.readyState === 'complete') {
    boot();
  } else {
    window.addEventListener('load', boot);
  }

})();
