import { Component, OnInit, inject } from '@angular/core';
import { NgFor, NgClass } from '@angular/common';
import { NgApexchartsModule } from 'ng-apexcharts';
import { MockDataService, KpiCard } from '../../services/mock-data.service';
import { KpiCardComponent } from '../../shared/components/kpi-card/kpi-card.component';

@Component({
  selector: 'app-insurance',
  standalone: true,
  imports: [NgFor, NgClass, NgApexchartsModule, KpiCardComponent],
  templateUrl: './insurance.component.html',
})
export class InsuranceComponent implements OnInit {
  private svc = inject(MockDataService);

  kpis: KpiCard[] = [];
  strat = this.svc.riskStratification();
  ibnrTriangle = this.svc.ibnrTriangle();
  years = ['2022','2023','2024','2025','2026'];
  devPeriods = ['0','12m','24m','36m','48m'];

  lossRatioChartOptions: any = {};
  pmpmChartOptions: any = {};
  ibnrChartOptions: any = {};
  stratChartOptions: any = {};

  ngOnInit(): void {
    this.kpis = this.svc.insuranceKpis();
    this.buildLossRatioChart();
    this.buildPmpmChart();
    this.buildIbnrChart();
    this.buildStratChart();
  }

  private buildLossRatioChart(): void {
    this.lossRatioChartOptions = {
      series: [
        { name: 'HealthRisk AI', data: [0.97, 0.98, 1.01, 0.99, 1.02, 1.00, 0.98, 1.03, 0.99, 1.01] },
        { name: 'GLM Baseline',  data: [0.45, 0.62, 0.78, 0.91, 1.00, 1.08, 1.16, 1.22, 1.28, 1.31] },
      ],
      chart: { type: 'line', height: 260, background: 'transparent', toolbar: { show: false } },
      stroke: { curve: 'smooth', width: [3, 2] },
      colors: ['#34a853','#ea4335'],
      xaxis: { categories: ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10'],
               title: { text: 'Cost Decile', style: { color: '#8b949e' } },
               labels: { style: { colors: '#8b949e' } } },
      yaxis: { min: 0.3, max: 1.4,
               labels: { formatter: (v: number) => v.toFixed(2), style: { colors: '#8b949e' } } },
      annotations: { yaxis: [
        { y: 0.95, y2: 1.05, fillColor: '#34a853', opacity: 0.08,
          label: { text: 'Target band ±5%', style: { color: '#34a853', background: 'transparent' } } }
      ] },
      tooltip: { theme: 'dark' },
      legend: { labels: { colors: '#8b949e' } },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      theme: { mode: 'dark' },
    };
  }

  private buildPmpmChart(): void {
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    this.pmpmChartOptions = {
      series: [
        { name: 'Actual PMPM',    data: [840,852,858,865,871,878,882,891,896,902,908,914] },
        { name: 'Predicted PMPM', data: [838,855,861,863,874,875,885,888,898,900,912,916] },
        { name: 'Budget',         data: Array(12).fill(880) },
      ],
      chart: { type: 'area', height: 260, background: 'transparent', toolbar: { show: false } },
      stroke: { curve: 'smooth', width: [2,2,1], dashArray: [0,0,4] },
      colors: ['#1a73e8','#34a853','#8b949e'],
      fill: { type: ['gradient','none','none'], gradient: { opacityFrom: 0.2, opacityTo: 0.01 } },
      xaxis: { categories: months, labels: { style: { colors: '#8b949e', fontSize: '10px' } } },
      yaxis: { labels: { formatter: (v: number) => `$${v}`, style: { colors: '#8b949e' } } },
      tooltip: { theme: 'dark', y: { formatter: (v: number) => `$${v}/member` } },
      legend: { labels: { colors: '#8b949e' } },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      theme: { mode: 'dark' },
    };
  }

  private buildIbnrChart(): void {
    this.ibnrChartOptions = {
      series: [{ name: 'IBNR Reserve', data: [18.2, 19.8, 21.4, 22.9, 24.1] }],
      chart: { type: 'bar', height: 200, background: 'transparent', toolbar: { show: false } },
      colors: ['#a855f7'],
      xaxis: { categories: ['Q3 2025','Q4 2025','Q1 2026','Q2 2026','Q3 2026 (est)'],
               labels: { style: { colors: '#8b949e', fontSize: '10px' } } },
      yaxis: { labels: { formatter: (v: number) => `$${v}M`, style: { colors: '#8b949e' } } },
      plotOptions: { bar: { borderRadius: 4 } },
      tooltip: { theme: 'dark', y: { formatter: (v: number) => `$${v}M` } },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      theme: { mode: 'dark' },
    };
  }

  private buildStratChart(): void {
    this.stratChartOptions = {
      series: [{ name: 'Members', data: this.strat.map(s => s.count) }],
      chart: { type: 'bar', height: 200, background: 'transparent', toolbar: { show: false } },
      colors: ['#ea4335','#f77f00','#f9c74f','#34a853'],
      xaxis: { categories: this.strat.map(s => s.tier),
               labels: { style: { colors: '#8b949e', fontSize: '10px' } } },
      yaxis: { labels: { formatter: (v: number) => v.toLocaleString(), style: { colors: '#8b949e' } } },
      plotOptions: { bar: { borderRadius: 4, distributed: true } },
      legend: { show: false },
      tooltip: { theme: 'dark' },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      theme: { mode: 'dark' },
    };
  }
}
