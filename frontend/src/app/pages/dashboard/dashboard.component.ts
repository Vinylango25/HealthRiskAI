import { Component, OnInit, inject, signal } from '@angular/core';
import { NgFor, NgClass, NgIf, DecimalPipe } from '@angular/common';
import { NgApexchartsModule } from 'ng-apexcharts';
import { MockDataService, KpiCard } from '../../services/mock-data.service';
import { KpiCardComponent } from '../../shared/components/kpi-card/kpi-card.component';
import { ApiService, LiveDashboard } from '../../services/api.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [NgFor, NgClass, NgIf, DecimalPipe, NgApexchartsModule, KpiCardComponent],
  templateUrl: './dashboard.component.html',
  styleUrl: './dashboard.component.scss',
})
export class DashboardComponent implements OnInit {
  private svc  = inject(MockDataService);
  private api  = inject(ApiService);

  kpis: KpiCard[] = [];
  epidemic: { country: string; r0: number; growth: number; alert: boolean }[] = [];

  // Live data from real API
  live = signal<LiveDashboard | null>(null);
  liveLoading = signal(true);
  apiStatus = signal<'live' | 'mock' | 'loading'>('loading');

  portfolioChartOptions: any = {};
  allocationChartOptions: any = {};
  riskGaugeOptions: any = {};

  ngOnInit(): void {
    this.kpis      = this.svc.portfolioKpis();
    this.epidemic  = this.svc.epidemicData();

    // Pull live data — updates KPIs that have real equivalents
    this.api.liveDashboard().subscribe(data => {
      this.live.set(data);
      this.liveLoading.set(false);
      this.apiStatus.set(data.fallback ? 'mock' : 'live');

      // Patch KPIs with real values where available
      if (!data.fallback) {
        this.kpis = this.kpis.map(k => {
          if (k.id === 'horizon' && data.recruiting_trials) {
            return { ...k, value: data.recruiting_trials.toLocaleString(), unit: 'trials', label: 'Recruiting Trials', icon: '🔬', description: 'Active recruiting trials on ClinicalTrials.gov', delta: 'ClinicalTrials.gov live' };
          }
          return k;
        });
      }
    });
    this.buildPortfolioChart();
    this.buildAllocationChart();
    this.buildRiskGauge();
  }

  private buildPortfolioChart(): void {
    const curves = this.svc.portfolioEquityCurve();
    this.portfolioChartOptions = {
      series: [
        { name: 'AI Model Portfolio', data: curves[0].map(p => p.y), color: '#1a73e8' },
        { name: 'Player Portfolio',   data: curves[1].map(p => p.y), color: '#34a853' },
        { name: 'Benchmark (+6%/yr)',  data: curves[2].map(p => p.y), color: '#8b949e' },
      ],
      chart: { type: 'line', height: 280, background: 'transparent', toolbar: { show: false },
               animations: { enabled: true, speed: 800 } },
      stroke: { curve: 'smooth', width: [3, 2, 1.5], dashArray: [0, 0, 4] },
      xaxis: { categories: curves[0].map(p => p.x),
               labels: { style: { colors: '#8b949e', fontSize: '10px' }, rotate: 0,
                         formatter: (v: string) => v.endsWith('0') || v === 'Q0' || parseInt(v.replace('Q','')) % 8 === 0 ? v : '' } },
      yaxis: { labels: { style: { colors: '#8b949e', fontSize: '10px' },
               formatter: (v: number) => `$${v.toFixed(0)}M` } },
      tooltip: { theme: 'dark', y: { formatter: (v: number) => `$${v.toFixed(1)}M` } },
      legend: { labels: { colors: '#8b949e' }, fontSize: '11px' },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      annotations: {
        xaxis: [{ x: 'Q8', borderColor: '#ea4335', label: { text: 'Pandemic', style: { color: '#ea4335', background: '#21262d', fontSize: '10px' } } }]
      },
      theme: { mode: 'dark' },
    };
  }

  private buildAllocationChart(): void {
    this.allocationChartOptions = {
      series: [40, 28, 22, 10],
      chart: { type: 'donut', height: 260, background: 'transparent' },
      labels: ['Insurance Book', 'Hospital Bonds', 'Pharma Equities', 'Leveraged Loans'],
      colors: ['#1a73e8', '#34a853', '#a855f7', '#f77f00'],
      legend: { labels: { colors: '#8b949e' }, fontSize: '11px', position: 'bottom' },
      plotOptions: { pie: { donut: {
        size: '65%',
        labels: { show: true, total: { show: true, label: 'Total AUM', color: '#8b949e',
          formatter: () => '$500M' } }
      } } },
      tooltip: { theme: 'dark' },
      dataLabels: { style: { fontSize: '11px' } },
      theme: { mode: 'dark' },
    };
  }

  private buildRiskGauge(): void {
    this.riskGaugeOptions = {
      series: [62],
      chart: { type: 'radialBar', height: 260, background: 'transparent' },
      plotOptions: { radialBar: {
        startAngle: -135, endAngle: 135,
        hollow: { size: '60%' },
        dataLabels: {
          name: { fontSize: '13px', color: '#8b949e', offsetY: -10 },
          value: { fontSize: '28px', fontWeight: 700, color: '#e6edf3',
                   formatter: (v: number) => `${v}/100` },
        },
        track: { background: '#30363d' },
      } },
      fill: { type: 'gradient', gradient: { shade: 'dark', type: 'horizontal',
        gradientToColors: ['#ea4335'], stops: [0, 100],
        colorStops: [
          { offset: 0, color: '#34a853', opacity: 1 },
          { offset: 50, color: '#f9c74f', opacity: 1 },
          { offset: 100, color: '#ea4335', opacity: 1 },
        ] } },
      labels: ['Risk Score'],
      theme: { mode: 'dark' },
    };
  }
}
