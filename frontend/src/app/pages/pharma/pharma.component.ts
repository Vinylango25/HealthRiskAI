import { Component, OnInit, inject } from '@angular/core';
import { NgFor, NgClass } from '@angular/common';
import { NgApexchartsModule } from 'ng-apexcharts';
import { MockDataService, KpiCard } from '../../services/mock-data.service';
import { KpiCardComponent } from '../../shared/components/kpi-card/kpi-card.component';

@Component({
  selector: 'app-pharma',
  standalone: true,
  imports: [NgFor, NgClass, NgApexchartsModule, KpiCardComponent],
  templateUrl: './pharma.component.html',
})
export class PharmaComponent implements OnInit {
  private svc = inject(MockDataService);
  kpis: KpiCard[] = [];

  rnpvChartOptions: any = {};
  cliffChartOptions: any = {};
  pipelineChartOptions: any = {};

  pipeline = [
    { asset:'PH-01', indication:'Oncology',   phase:'Phase III', enrollment:88, rnpv:142, status:'on-track' },
    { asset:'PH-02', indication:'Cardiology', phase:'Phase III', enrollment:73, rnpv: 98, status:'at-risk'  },
    { asset:'PH-03', indication:'Oncology',   phase:'Phase III', enrollment:91, rnpv:186, status:'on-track' },
    { asset:'PH-04', indication:'CNS',        phase:'Phase II',  enrollment:65, rnpv: 54, status:'at-risk'  },
    { asset:'PH-05', indication:'Autoimmune', phase:'Phase II',  enrollment:82, rnpv: 71, status:'on-track' },
    { asset:'PH-06', indication:'Rare',       phase:'Phase I',   enrollment:95, rnpv: 28, status:'on-track' },
    { asset:'PH-07', indication:'Oncology',   phase:'Phase I',   enrollment:44, rnpv: 19, status:'safety-flag'},
  ];

  ngOnInit(): void {
    this.kpis = this.svc.pharmaKpis();
    this.buildRnpvChart();
    this.buildCliffChart();
    this.buildPipelineChart();
  }

  private buildRnpvChart(): void {
    // rNPV Monte Carlo distribution
    const bins = Array.from({length: 30}, (_, i) => -350 + i * 30);
    const dist = [0,0,1,2,4,7,11,16,20,22,18,14,10,7,5,4,3,2,2,1,1,1,0,0,0,0,0,0,0,0];
    this.rnpvChartOptions = {
      series: [{ name: 'Scenarios', data: dist }],
      chart: { type: 'bar', height: 250, background: 'transparent', toolbar: { show: false } },
      colors: ['#1a73e8'],
      xaxis: { categories: bins.map(b => `$${b}M`),
               labels: { rotate: -45, style: { colors: '#8b949e', fontSize: '9px' } } },
      yaxis: { labels: { style: { colors: '#8b949e' } } },
      plotOptions: { bar: { borderRadius: 2, columnWidth: '90%' } },
      annotations: { xaxis: [
        { x: '$140M', borderColor: '#34a853', label: { text: 'Mean rNPV $142M', style: { color: '#34a853', background: 'transparent' } } }
      ] },
      tooltip: { theme: 'dark', y: { formatter: (v: number) => `${v} scenarios` } },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      theme: { mode: 'dark' },
    };
  }

  private buildCliffChart(): void {
    const years = ['Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Y10'];
    const sm  = [100,100,100,52,28,18,12,10,9,8];
    const bio = [100,100,100,90,78,65,55,46,40,36];
    this.cliffChartOptions = {
      series: [
        { name: 'Small Molecule (patent cliff)', data: sm },
        { name: 'Biologic (biosimilar delay)', data: bio },
      ],
      chart: { type: 'area', height: 250, background: 'transparent', toolbar: { show: false } },
      stroke: { curve: 'smooth', width: [2,2] },
      colors: ['#ea4335','#34a853'],
      fill: { type: 'gradient', gradient: { opacityFrom: 0.25, opacityTo: 0.02 } },
      xaxis: { categories: years, title: { text: 'Years post-patent expiry', style: { color: '#8b949e' } },
               labels: { style: { colors: '#8b949e' } } },
      yaxis: { max: 110, labels: { formatter: (v: number) => `${v}%`, style: { colors: '#8b949e' } } },
      annotations: { yaxis: [{ y: 50, borderColor: '#8b949e', label: { text: '50% threshold', style: { color: '#8b949e', background: 'transparent' } } }] },
      tooltip: { theme: 'dark', y: { formatter: (v: number) => `${v}% of peak revenue` } },
      legend: { labels: { colors: '#8b949e' } },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      theme: { mode: 'dark' },
    };
  }

  private buildPipelineChart(): void {
    this.pipelineChartOptions = {
      series: [{ name: 'Enrollment %', data: this.pipeline.map(p => p.enrollment) }],
      chart: { type: 'bar', height: 220, background: 'transparent', toolbar: { show: false } },
      colors: this.pipeline.map(p => p.status === 'on-track' ? '#34a853' : p.status === 'at-risk' ? '#f77f00' : '#ea4335'),
      xaxis: { categories: this.pipeline.map(p => p.asset), labels: { style: { colors: '#8b949e' } } },
      yaxis: { max: 100, labels: { formatter: (v: number) => `${v}%`, style: { colors: '#8b949e' } } },
      plotOptions: { bar: { borderRadius: 4, distributed: true } },
      annotations: { yaxis: [{ y: 80, borderColor: '#f9c74f', borderWidth: 1,
        label: { text: '80% minimum', style: { color: '#f9c74f', background: 'transparent', fontSize: '10px' } } }] },
      legend: { show: false },
      tooltip: { theme: 'dark', y: { formatter: (v: number) => `${v}% enrolled` } },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      theme: { mode: 'dark' },
    };
  }

  statusColor(s: string): string {
    return s === 'on-track' ? 'green' : s === 'at-risk' ? 'orange' : 'red';
  }
}
