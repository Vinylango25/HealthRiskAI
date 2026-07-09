import { Component, OnInit, inject } from '@angular/core';
import { NgFor, NgClass } from '@angular/common';
import { NgApexchartsModule } from 'ng-apexcharts';
import { MockDataService } from '../../services/mock-data.service';

@Component({
  selector: 'app-explainability',
  standalone: true,
  imports: [NgFor, NgClass, NgApexchartsModule],
  templateUrl: './explainability.component.html',
})
export class ExplainabilityComponent implements OnInit {
  private svc = inject(MockDataService);
  Math = Math;

  shapValues = this.svc.shapValues();
  waterfallOptions: any = {};
  globalShapOptions: any = {};
  pdpAgeOptions: any = {};
  pdpHba1cOptions: any = {};

  // Counterfactual patient
  patient = { age: 72, priorAdmissions: 2, hba1c: 8.6, creatinine: 1.9, erVisits: 3, hcc: 2.4 };
  counterfactual = { age: 72, priorAdmissions: 1, hba1c: 7.0, creatinine: 1.4, erVisits: 1, hcc: 2.4 };
  patientRisk = 0.57;
  counterfactualRisk = 0.28;

  ngOnInit(): void {
    this.buildWaterfall();
    this.buildGlobalShap();
    this.buildPdpAge();
    this.buildPdpHba1c();
  }

  private buildWaterfall(): void {
    const contributions = [
      { label: 'Prior Admissions', value: 0.142 },
      { label: 'HCC Score', value: 0.091 },
      { label: 'Age 72', value: 0.071 },
      { label: 'ER Visits (3)', value: 0.058 },
      { label: 'HbA1c 8.6%', value: 0.044 },
      { label: 'Medication Adherence', value: -0.038 },
      { label: 'No Prior Surgery', value: -0.019 },
    ];
    const colors = contributions.map(c => c.value > 0 ? '#ea4335' : '#34a853');
    this.waterfallOptions = {
      series: [{ name: 'SHAP Value', data: contributions.map(c => Math.abs(c.value)) }],
      chart: { type: 'bar', height: 280, background: 'transparent', toolbar: { show: false } },
      colors,
      plotOptions: { bar: { horizontal: true, borderRadius: 3, distributed: true } },
      xaxis: { labels: { formatter: (v: number) => v.toFixed(3), style: { colors: '#8b949e' } } },
      yaxis: { categories: contributions.map(c => c.label), labels: { style: { colors: '#8b949e', fontSize: '11px' } } },
      tooltip: { theme: 'dark', y: { formatter: (v: number, opts: any) => {
        const c = contributions[opts.dataPointIndex];
        return `${c.value > 0 ? '+' : ''}${c.value.toFixed(3)} (${c.value > 0 ? 'increases' : 'decreases'} risk)`;
      } } },
      legend: { show: false },
      grid: { borderColor: '#30363d' },
      theme: { mode: 'dark' },
    };
  }

  private buildGlobalShap(): void {
    this.globalShapOptions = {
      series: [{ name: 'Mean |SHAP|', data: this.shapValues.map(s => Math.abs(s.value)) }],
      chart: { type: 'bar', height: 280, background: 'transparent', toolbar: { show: false } },
      colors: ['#1a73e8'],
      plotOptions: { bar: { horizontal: true, borderRadius: 3 } },
      xaxis: { labels: { formatter: (v: number) => v.toFixed(3), style: { colors: '#8b949e' } } },
      yaxis: { categories: this.shapValues.map(s => s.feature), labels: { style: { colors: '#8b949e', fontSize: '11px' } } },
      annotations: { xaxis: [{ x: 0.06, borderColor: '#ea4335', borderWidth: 1,
        label: { text: 'Significance threshold', style: { color: '#ea4335', background: 'transparent', fontSize: '9px' } } }] },
      tooltip: { theme: 'dark', y: { formatter: (v: number) => v.toFixed(3) } },
      grid: { borderColor: '#30363d' },
      theme: { mode: 'dark' },
    };
  }

  private buildPdpAge(): void {
    const data = this.svc.pdpAgeData();
    this.pdpAgeOptions = {
      series: [{ name: 'Readmission Risk', data: data.map(d => ({ x: d.x, y: d.y })) }],
      chart: { type: 'line', height: 220, background: 'transparent', toolbar: { show: false } },
      stroke: { curve: 'smooth', width: 2 },
      colors: ['#1a73e8'],
      xaxis: { type: 'numeric', title: { text: 'Age (years)', style: { color: '#8b949e' } }, labels: { style: { colors: '#8b949e' } } },
      yaxis: { max: 0.9, labels: { formatter: (v: number) => `${(v*100).toFixed(0)}%`, style: { colors: '#8b949e' } } },
      annotations: { xaxis: [{ x: 65, borderColor: '#f9c74f', label: { text: 'Age 65', style: { color: '#f9c74f', background: 'transparent' } } }] },
      tooltip: { theme: 'dark', y: { formatter: (v: number) => `${(v*100).toFixed(1)}% risk` } },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      theme: { mode: 'dark' },
    };
  }

  private buildPdpHba1c(): void {
    const data = this.svc.pdpHba1cData();
    this.pdpHba1cOptions = {
      series: [{ name: 'Readmission Risk', data: data.map(d => ({ x: d.x, y: d.y })) }],
      chart: { type: 'line', height: 220, background: 'transparent', toolbar: { show: false } },
      stroke: { curve: 'smooth', width: 2 },
      colors: ['#a855f7'],
      xaxis: { type: 'numeric', title: { text: 'HbA1c (%)', style: { color: '#8b949e' } }, labels: { style: { colors: '#8b949e' } } },
      yaxis: { max: 0.95, labels: { formatter: (v: number) => `${(v*100).toFixed(0)}%`, style: { colors: '#8b949e' } } },
      annotations: { xaxis: [{ x: 7.0, borderColor: '#f9c74f', label: { text: 'ADA Target 7.0%', style: { color: '#f9c74f', background: 'transparent' } } }] },
      tooltip: { theme: 'dark', y: { formatter: (v: number) => `${(v*100).toFixed(1)}% risk` } },
      grid: { borderColor: '#30363d', strokeDashArray: 3 },
      theme: { mode: 'dark' },
    };
  }
}
