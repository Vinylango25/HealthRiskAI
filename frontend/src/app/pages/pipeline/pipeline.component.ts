import { Component, inject, signal } from '@angular/core';
import { NgFor, NgClass } from '@angular/common';
import { MockDataService } from '../../services/mock-data.service';

@Component({
  selector: 'app-pipeline',
  standalone: true,
  imports: [NgFor, NgClass],
  templateUrl: './pipeline.component.html',
  styleUrl: './pipeline.component.scss',
})
export class PipelineComponent {
  private svc = inject(MockDataService);

  pipelines = signal(this.svc.pipelineStatus());
  runningId = signal<string | null>(null);
  logs = signal<string[]>([
    '[14:02:31] MIMIC-IV Data Ingestion — COMPLETE (4m 23s)',
    '[14:06:44] WHO GHO Fetch — COMPLETE (1m 12s)',
    '[17:30:01] FDA FAERS Update — IN PROGRESS (step 3/5: deduplication)',
    '[17:30:18] Processing 1,240,382 adverse event records...',
    '[17:34:21] SHAP Explainability — FAILED: Memory limit exceeded at 45%',
  ]);

  statusColor(s: string): string {
    return s === 'success' ? 'green' : s === 'running' ? 'blue' : s === 'failed' ? 'red' : s === 'queued' ? 'yellow' : 'orange';
  }

  statusIcon(s: string): string {
    return s === 'success' ? '✅' : s === 'running' ? '⚙️' : s === 'failed' ? '❌' : s === 'queued' ? '⏳' : '⚠️';
  }

  runPipeline(id: string): void {
    this.runningId.set(id);
    this.logs.update(l => [...l, `[${new Date().toLocaleTimeString()}] Starting pipeline ${id}...`]);
    const pls = this.pipelines();
    const idx = pls.findIndex(p => p.id === id);
    if (idx !== -1) {
      const updated = [...pls];
      updated[idx] = { ...updated[idx], status: 'running', progress: 0 };
      this.pipelines.set(updated);
    }

    // Simulate progress
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.floor(Math.random() * 15) + 5;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        const done = this.pipelines();
        const i = done.findIndex(p => p.id === id);
        if (i !== -1) {
          const u = [...done];
          u[i] = { ...u[i], status: 'success', progress: 100, lastRun: new Date().toLocaleString(), duration: '~2m' };
          this.pipelines.set(u);
        }
        this.runningId.set(null);
        this.logs.update(l => [...l, `[${new Date().toLocaleTimeString()}] Pipeline ${id} — COMPLETE`]);
      } else {
        const mid = this.pipelines();
        const i = mid.findIndex(p => p.id === id);
        if (i !== -1) {
          const u = [...mid];
          u[i] = { ...u[i], progress };
          this.pipelines.set(u);
        }
        this.logs.update(l => [...l, `[${new Date().toLocaleTimeString()}] ${id}: ${progress}% complete`]);
      }
    }, 600);
  }
}
