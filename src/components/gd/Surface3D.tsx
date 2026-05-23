import { Canvas, type ThreeEvent } from "@react-three/fiber";
import { Line, OrbitControls } from "@react-three/drei";
import { useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { LossSpec, Trajectory, Vec2 } from "../../lib/gradientDescent";

interface Surface3DProps {
  loss: LossSpec;
  start: Vec2;
  trajectories: Trajectory[];
  onSetStart: (p: Vec2) => void;
}

const HALF_W = 1.5;
const TARGET_HEIGHT = 1.4;
const TRAJ_OFFSET = 0.02;

export default function Surface3D({
  loss,
  start,
  trajectories,
  onSetStart,
}: Surface3DProps) {
  const [wireframe, setWireframe] = useState(false);
  return (
    <div className="rounded-2xl bg-ink-950/60 border border-ink-800/80 overflow-hidden flex flex-col">
      <div className="flex items-center justify-between text-[11px] px-3 py-2 border-b border-ink-800/80">
        <span className="text-[10px] uppercase tracking-wider text-ink-400 font-bold">
          Loss landscape · 3D
        </span>
        <span className="flex items-center gap-3">
          <button
            onClick={() => setWireframe((w) => !w)}
            className={`px-2 py-0.5 rounded text-[10.5px] font-mono border transition-colors ${
              wireframe
                ? "border-emerald-400/60 text-white bg-emerald-500/10"
                : "border-ink-700 text-ink-400 hover:text-white"
            }`}
          >
            wireframe
          </button>
          <span className="text-ink-500 font-mono hidden sm:inline">
            drag · rotate &nbsp; · &nbsp; scroll · zoom &nbsp; · &nbsp; click surface · set start
          </span>
        </span>
      </div>
      <div className="h-[520px] relative">
        <Canvas
          camera={{ position: [2.6, 2.2, 2.6], fov: 45 }}
          dpr={[1, 2]}
          gl={{ antialias: true }}
        >
          <color attach="background" args={["#0b0d12"]} />
          <ambientLight intensity={0.55} />
          <directionalLight position={[4, 6, 3]} intensity={0.9} />
          <directionalLight position={[-3, 3, -2]} intensity={0.35} color="#7c5cff" />
          <Scene
            loss={loss}
            start={start}
            trajectories={trajectories}
            onSetStart={onSetStart}
            wireframe={wireframe}
          />
          <OrbitControls
            enablePan={false}
            minDistance={2.2}
            maxDistance={9}
            maxPolarAngle={Math.PI / 2.05}
            target={[0, 0.25, 0]}
          />
        </Canvas>
      </div>
    </div>
  );
}

/* =============================== scene =============================== */

function Scene({
  loss,
  start,
  trajectories,
  onSetStart,
  wireframe,
}: Surface3DProps & { wireframe: boolean }) {
  const transform = useMemo(() => makeTransform(loss), [loss]);
  const geometry = useMemo(() => buildSurface(loss, transform, 90), [loss, transform]);

  const handleClick = (e: ThreeEvent<MouseEvent>) => {
    e.stopPropagation();
    const { x, z } = e.point;
    const ls = transform.toLoss(x, z);
    onSetStart([+ls[0].toFixed(3), +ls[1].toFixed(3)]);
  };

  return (
    <group>
      {/* Floor reference frame */}
      <gridHelper args={[3, 12, "#1f2330", "#161922"]} position={[0, -0.001, 0]} />

      {/* Surface */}
      <mesh
        geometry={geometry}
        onClick={handleClick}
        castShadow={false}
        receiveShadow={false}
      >
        <meshStandardMaterial
          vertexColors
          flatShading={false}
          metalness={0.05}
          roughness={0.85}
          side={THREE.DoubleSide}
          wireframe={wireframe}
        />
      </mesh>

      {/* Minima */}
      {loss.minima.map((m, idx) => {
        const z = loss.f(m);
        const [wx, wy, wz] = transform.toWorld(m[0], m[1], z);
        return (
          <group key={`min-${idx}`} position={[wx, wy + TRAJ_OFFSET, wz]}>
            <mesh>
              <sphereGeometry args={[0.04, 24, 24]} />
              <meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.4} />
            </mesh>
            <mesh>
              <ringGeometry args={[0.07, 0.085, 32]} />
              <meshBasicMaterial color="#ffffff" transparent opacity={0.5} side={THREE.DoubleSide} />
            </mesh>
          </group>
        );
      })}

      {/* Start marker */}
      <StartMarker loss={loss} start={start} transform={transform} />

      {/* Trajectories */}
      {trajectories.map((traj) => (
        <TrajectoryLine key={traj.optId} traj={traj} loss={loss} transform={transform} />
      ))}
    </group>
  );
}

/* =============================== markers =============================== */

function StartMarker({
  loss,
  start,
  transform,
}: {
  loss: LossSpec;
  start: Vec2;
  transform: Transform;
}) {
  const z = loss.f(start);
  const [wx, wy, wz] = transform.toWorld(start[0], start[1], z);
  return (
    <group position={[wx, wy + TRAJ_OFFSET, wz]}>
      <mesh>
        <sphereGeometry args={[0.05, 24, 24]} />
        <meshStandardMaterial color="#fff" emissive="#fff" emissiveIntensity={0.2} />
      </mesh>
      <mesh>
        <ringGeometry args={[0.085, 0.105, 32]} />
        <meshBasicMaterial color="#ffffff" transparent opacity={0.6} side={THREE.DoubleSide} />
      </mesh>
      {/* vertical drop line */}
      <Line
        points={[
          [0, -wy, 0],
          [0, 0, 0],
        ]}
        color="#ffffff"
        transparent
        opacity={0.25}
        lineWidth={1}
        dashed
        dashScale={20}
        gapSize={0.05}
      />
    </group>
  );
}

function TrajectoryLine({
  traj,
  loss,
  transform,
}: {
  traj: Trajectory;
  loss: LossSpec;
  transform: Transform;
}) {
  const points = useMemo(() => {
    const out: [number, number, number][] = [];
    for (const p of traj.points) {
      const z = loss.f(p);
      const [wx, wy, wz] = transform.toWorld(p[0], p[1], z);
      out.push([wx, wy + TRAJ_OFFSET, wz]);
    }
    return out;
  }, [traj, loss, transform]);

  const groundPoints = useMemo(
    () => points.map(([x, , z]) => [x, 0.001, z] as [number, number, number]),
    [points]
  );

  const endRef = useRef<THREE.Mesh>(null);

  if (points.length < 2) return null;

  return (
    <group>
      {/* 3D path on surface */}
      <Line points={points} color={traj.color} lineWidth={3} />
      {/* shadow on floor */}
      <Line points={groundPoints} color={traj.color} lineWidth={1.2} transparent opacity={0.35} />
      {/* end-point marker */}
      <mesh ref={endRef} position={points[points.length - 1]}>
        <sphereGeometry args={[0.045, 18, 18]} />
        <meshStandardMaterial
          color={traj.color}
          emissive={traj.color}
          emissiveIntensity={0.45}
        />
      </mesh>
    </group>
  );
}

/* =============================== math: transform + geometry =============================== */

interface Transform {
  toWorld: (x: number, y: number, z: number) => [number, number, number];
  toLoss: (wx: number, wz: number) => Vec2;
  heightOf: (z: number) => number;
  zMin: number;
  zMax: number;
}

function makeTransform(loss: LossSpec): Transform {
  const { xMin, xMax, yMin, yMax } = loss.domain;
  // sample to find z range
  const N = 50;
  let zMin = Infinity;
  let zMax = -Infinity;
  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= N; j++) {
      const x = xMin + ((xMax - xMin) * j) / N;
      const y = yMin + ((yMax - yMin) * i) / N;
      const z = loss.f([x, y]);
      if (z < zMin) zMin = z;
      if (z > zMax) zMax = z;
    }
  }
  const zRange = Math.max(zMax - zMin, 1e-6);
  // log compression so steep cliffs (Rosenbrock) don't dominate
  const denom = Math.log1p(zRange) || 1;
  const heightOf = (z: number) => {
    const shifted = Math.max(z - zMin, 0);
    return (Math.log1p(shifted) / denom) * TARGET_HEIGHT;
  };
  const toWorld = (x: number, y: number, z: number): [number, number, number] => {
    const wx = (((x - xMin) / (xMax - xMin)) * 2 - 1) * HALF_W;
    const wz = -(((y - yMin) / (yMax - yMin)) * 2 - 1) * HALF_W;
    const wy = heightOf(z);
    return [wx, wy, wz];
  };
  const toLoss = (wx: number, wz: number): Vec2 => {
    const x = ((wx / HALF_W + 1) / 2) * (xMax - xMin) + xMin;
    const y = ((-wz / HALF_W + 1) / 2) * (yMax - yMin) + yMin;
    return [
      Math.max(xMin, Math.min(xMax, x)),
      Math.max(yMin, Math.min(yMax, y)),
    ];
  };
  return { toWorld, toLoss, heightOf, zMin, zMax };
}

function buildSurface(loss: LossSpec, transform: Transform, N = 90): THREE.BufferGeometry {
  const { xMin, xMax, yMin, yMax } = loss.domain;
  const vCount = (N + 1) * (N + 1);
  const positions = new Float32Array(vCount * 3);
  const colors = new Float32Array(vCount * 3);
  const indices: number[] = [];

  for (let i = 0; i <= N; i++) {
    for (let j = 0; j <= N; j++) {
      const idx = i * (N + 1) + j;
      const x = xMin + ((xMax - xMin) * j) / N;
      const y = yMin + ((yMax - yMin) * i) / N;
      const z = loss.f([x, y]);
      const [wx, wy, wz] = transform.toWorld(x, y, z);
      positions[idx * 3] = wx;
      positions[idx * 3 + 1] = wy;
      positions[idx * 3 + 2] = wz;
      const t = Math.min(Math.max(wy / TARGET_HEIGHT, 0), 1);
      const [r, g, b] = colorRamp(t);
      colors[idx * 3] = r;
      colors[idx * 3 + 1] = g;
      colors[idx * 3 + 2] = b;
    }
  }

  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      const a = i * (N + 1) + j;
      const b = i * (N + 1) + j + 1;
      const c = (i + 1) * (N + 1) + j;
      const d = (i + 1) * (N + 1) + j + 1;
      indices.push(a, c, b, b, c, d);
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return geo;
}

/* =============================== color ramp =============================== */

const STOPS: [number, number, number][] = [
  [0.07, 0.07, 0.15], // deep navy
  [0.28, 0.18, 0.50], // violet
  [0.42, 0.42, 0.85], // periwinkle
  [0.20, 0.70, 0.80], // cyan
  [0.95, 0.80, 0.35], // amber
  [0.92, 0.40, 0.40], // rose
];

function colorRamp(t: number): [number, number, number] {
  const seg = Math.min(Math.max(t, 0), 1) * (STOPS.length - 1);
  const i = Math.floor(seg);
  const f = seg - i;
  const a = STOPS[Math.min(i, STOPS.length - 1)];
  const b = STOPS[Math.min(i + 1, STOPS.length - 1)];
  return [
    a[0] * (1 - f) + b[0] * f,
    a[1] * (1 - f) + b[1] * f,
    a[2] * (1 - f) + b[2] * f,
  ];
}
