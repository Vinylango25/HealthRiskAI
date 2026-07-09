import { Component, OnInit, inject } from '@angular/core';
import { NgFor, NgClass } from '@angular/common';
import { NgApexchartsModule } from 'ng-apexcharts';
import { MockDataService, KpiCard } from '../../services/mock-data.service';
import { KpiCardComponent } from '../../shared/components/kpi-card/kpi-card.component';

@Component({
  selector: 'app-credit-risk',
  standalone: true,
  imports: [NgFor, NgClass, NgApexchartsModule, KpiCardComponent],
  templateUrl: './credit-risk.component.html',
})
export class CreditRiskComponent implements OnInit {
  private svc = inject(MockDataService);

  kpis: KpiCard[] = [];
  hospitals = this.svc.hospitalScorecard();

  rocChartOptions: any = {};
  spreadChartOptions: any = {};
  scoreDistOptions: any = {};

  ngOnInit(): void {
    this.kpis = this.svc.creditRiskKpis();
    this.buildRocChart();
    this.buildSpreadChart();
    this.buildScoreDist();
  }

  private buildRocChart(): void {
    const pts = (auroc: number) => Array.from({length: 51}, (_, i) => {
      const fpr = i / 50;
      const tpr = Math.min(1, fpr + (auroc - 0.5) * 2 * Math.sqrt(fpr * (1 - fpr) + 0.001));
      return parseFloat(tpr.toFixed(3));
    });
    const fprs = Array.from({length: 51}, (_, i) => parseFloat((i/50).toFixed(2)));
    this.rocChartOptions = {
      series: [
        { name: 'HealthRisk AI (0.851)', data: fprs.map((x, i) => ({ x, y: pts(0.851)[i] })) },
        { name: 'Financial Only (0.742)', data: fprs.map((x, i) => ({ x, y: pts(0.742)[i] })) },
        { name: 'Diagonal', data: fprs.map(x => ({ x, y: x })) },
      ],
      chart: { type: 'line', height: 260, background: 'transparent', toolbar: { show: false } },
      stroke: { curve: 'straight', width: [3,2,1], dashArray: [0,0,4] },
      colors: ['#34a853','#f77f00','#30363d'],
      xaxis: { type: 'numeric', title: { text: 'False Positive Rate', style: { color: '#8b949e' } },
               labels: { style: { colors: '#8b949e' } }, min: 0, max: 1 },
      yaxis: { min: 0, max: 1, title: { text: 'True Positive Rate', style: { color: '#8b949e' } },
               labels: { style: { colors: '#8b949e' } } },
      tooltip: { theme: 'dark' },
      legend: { labels: { colors: '#8b949e' } },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      theme: { mode: 'dark' },
    };
  }

  private buildSpreadChart(): void {
    const curves = this.svc.bondSpreadHistory();
    this.spreadChartOptions = {
      series: [
        { name: 'Portfolio Avg (bps)', data: curves[0].map(p => p.y) },
        { name: 'Distressed (Oakridge)', data: curves[1].map(p => p.y) },
      ],
      chart: { type: 'area', height: 260, background: 'transparent', toolbar: { show: false } },
      stroke: { curve: 'smooth', width: [2,2] },
      colors: ['#1a73e8','#ea4335'],
      fill: { type: 'gradient', gradient: { opacityFrom: 0.15, opacityTo: 0.02 } },
      xaxis: { categories: curves[0].map(p => p.x), labels: { style: { colors: '#8b949e' } } },
      yaxis: { labels: { formatter: (v: number) => `${v} bps`, style: { colors: '#8b949e' } } },
      tooltip: { theme: 'dark', y: { formatter: (v: number) => `${v} bps` } },
      legend: { labels: { colors: '#8b949e' } },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      theme: { mode: 'dark' },
    };
  }

  private buildScoreDist(): void {
    const cats = Array.from({length: 20}, (_, i) => (i * 5).toString());
    const stable = [0,0,1,3,5,8,12,15,18,16,12,8,5,3,2,1,0,0,0,0];
    const watch  = [0,0,0,1,3,6,10,14,12,10,8,6,4,2,1,0,0,0,0,0];
    this.scoreDistOptions = {
      series: [
        { name: 'Stable (PD < 3%)',  data: stable },
        { name: 'Distressed (PD > 3%)', data: watch },
      ],
      chart: { type: 'bar', height: 220, background: 'transparent', toolbar: { show: false },
               stacked: false },
      colors: ['#34a853','#ea4335'],
      xaxis: { categories: cats, title: { text: 'Credit Score', style: { color: '#8b949e' } },
               labels: { style: { colors: '#8b949e', fontSize: '9px' } } },
      yaxis: { labels: { style: { colors: '#8b949e' } } },
      plotOptions: { bar: { borderRadius: 2 } },
      tooltip: { theme: 'dark' },
      legend: { labels: { colors: '#8b949e' } },
      grid: { borderColor: '#30363d' },
      theme: { mode: 'dark' },
    };
  }

  statusColor(s: string): string {
    return s === 'stable' ? 'green' : s === 'watch' ? 'yellow' : 'red';
  }
}
