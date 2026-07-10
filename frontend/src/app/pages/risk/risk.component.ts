import { Component, signal, OnInit, inject } from '@angular/core';
import { NgFor, NgClass, NgIf } from '@angular/common';
import { NgApexchartsModule } from 'ng-apexcharts';
import { MockDataService, KpiCard } from '../../services/mock-data.service';
import { KpiCardComponent } from '../../shared/components/kpi-card/kpi-card.component';

@Component({
  selector: 'app-risk',
  standalone: true,
  imports: [NgFor, NgClass, NgIf, NgApexchartsModule, KpiCardComponent],
  template: `
    <!-- ── Inner sub-tab bar ─────────────────────────────────────── -->
    <div class="inner-tabs">
      <button class="inner-tab" [class.inner-tab--active]="tab() === 'credit'"
        (click)="tab.set('credit')">🏦 Credit Risk</button>
      <button class="inner-tab" [class.inner-tab--active]="tab() === 'pharma'"
        (click)="tab.set('pharma')">💊 Pharma</button>
    </div>

    <!-- ═══════════════ CREDIT RISK ═══════════════ -->
    <ng-container *ngIf="tab() === 'credit'">
      <div class="page-header">
        <h1>Hospital Credit Risk</h1>
        <p>Hospital bond PD model — AUROC 0.851, Gini 0.702 · 15 issuers · $280M exposure</p>
      </div>

      <div class="kpi-grid">
        <app-kpi-card *ngFor="let k of creditKpis" [kpi]="k"></app-kpi-card>
      </div>

      <div class="alert-banner danger">
        <span class="alert-icon">🚨</span>
        <div class="alert-text">
          <strong>Oakridge Medical — Critical:</strong> PD risen to 6.8% (+62% in 90 days). CMI declining for 3 consecutive quarters. Recommend bond position review.
        </div>
        <span class="alert-time">2h ago</span>
      </div>

      <div class="section-grid cols-2" style="margin-bottom:20px">
        <div class="card">
          <div class="card__header">
            <span class="card__title">PD Model ROC Curves</span>
            <span class="badge green">AUROC 0.851</span>
          </div>
          <p style="font-size:11px;color:var(--text-muted);margin-bottom:8px">
            HealthRisk AI clinical+financial model achieves AUROC 0.851 vs 0.742 for financial-only.
          </p>
          <apx-chart [series]="rocChart.series" [chart]="rocChart.chart"
            [stroke]="rocChart.stroke" [colors]="rocChart.colors"
            [xaxis]="rocChart.xaxis" [yaxis]="rocChart.yaxis"
            [tooltip]="rocChart.tooltip" [legend]="rocChart.legend"
            [grid]="rocChart.grid" [theme]="rocChart.theme">
          </apx-chart>
        </div>
        <div class="card">
          <div class="card__header">
            <span class="card__title">Bond Spread History (bps over Treasury)</span>
            <span class="badge red">+12 bps QoQ</span>
          </div>
          <apx-chart [series]="spreadChart.series" [chart]="spreadChart.chart"
            [stroke]="spreadChart.stroke" [colors]="spreadChart.colors"
            [fill]="spreadChart.fill" [xaxis]="spreadChart.xaxis"
            [yaxis]="spreadChart.yaxis" [tooltip]="spreadChart.tooltip"
            [legend]="spreadChart.legend" [grid]="spreadChart.grid"
            [theme]="spreadChart.theme">
          </apx-chart>
        </div>
      </div>

      <div class="section-grid cols-2">
        <div class="card">
          <div class="card__header">
            <span class="card__title">Hospital Scorecard</span>
            <span class="badge yellow">4 Watchlist</span>
          </div>
          <div class="table-responsive-wrap">
            <table class="data-table">
              <thead><tr><th>Hospital</th><th>Score</th><th>PD</th><th>Spread</th><th>Status</th></tr></thead>
              <tbody>
                <tr *ngFor="let h of hospitals">
                  <td>{{ h.name }}</td>
                  <td>
                    <div class="flex items-center gap-8">
                      <span [class.text-green]="h.score>=65" [class.text-yellow]="h.score>=50&&h.score<65" [class.text-red]="h.score<50">{{ h.score }}</span>
                      <div class="progress-bar" style="width:80px">
                        <div class="progress-bar__fill"
                          [style.width]="h.score+'%'"
                          [style.background]="h.score>=65?'var(--green)':h.score>=50?'var(--yellow)':'var(--red)'">
                        </div>
                      </div>
                    </div>
                  </td>
                  <td [class.text-red]="h.pd>5" [class.text-yellow]="h.pd>2&&h.pd<=5" [class.text-green]="h.pd<=2">{{ h.pd }}%</td>
                  <td>{{ h.spread }} bps</td>
                  <td><span class="badge" [ngClass]="creditColor(h.status)">{{ h.status }}</span></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        <div class="card">
          <div class="card__header">
            <span class="card__title">Score Distribution — HealthRisk AI vs Financial-Only</span>
          </div>
          <apx-chart [series]="scoreDistChart.series" [chart]="scoreDistChart.chart"
            [colors]="scoreDistChart.colors" [xaxis]="scoreDistChart.xaxis"
            [yaxis]="scoreDistChart.yaxis" [plotOptions]="scoreDistChart.plotOptions"
            [tooltip]="scoreDistChart.tooltip" [legend]="scoreDistChart.legend"
            [grid]="scoreDistChart.grid" [theme]="scoreDistChart.theme">
          </apx-chart>
          <div class="metric-row">
            <div class="metric"><span class="metric__label">Gini Coefficient</span><span class="metric__value text-green">0.702</span></div>
            <div class="metric"><span class="metric__label">KS Statistic</span><span class="metric__value text-blue">0.421</span></div>
            <div class="metric"><span class="metric__label">Avg ECL</span><span class="metric__value">$6.4M</span></div>
          </div>
        </div>
      </div>
    </ng-container>

    <!-- ═══════════════ PHARMA ═══════════════ -->
    <ng-container *ngIf="tab() === 'pharma'">
      <div class="page-header">
        <h1>Pharma / Biotech Analytics</h1>
        <p>rNPV Monte Carlo · Patent cliff model · Pipeline enrollment monitor · FAERS safety signals</p>
      </div>

      <div class="kpi-grid">
        <app-kpi-card *ngFor="let k of pharmaKpis" [kpi]="k"></app-kpi-card>
      </div>

      <div class="alert-banner warning">
        <span class="alert-icon">⚠️</span>
        <div class="alert-text">
          <strong>FAERS Safety Signal — PH-07:</strong> Disproportionality ROR = 3.4 (threshold 2.0). Adverse event category: hepatotoxicity.
        </div>
      </div>

      <div class="section-grid cols-2" style="margin-bottom:20px">
        <div class="card">
          <div class="card__header">
            <span class="card__title">rNPV Monte Carlo Distribution (Phase III Oncology)</span>
            <span class="badge blue">Mean $142M</span>
          </div>
          <p style="font-size:11px;color:var(--text-muted);margin-bottom:8px">
            5,000-scenario simulation. Left peak = Phase III failure (−$313M). P(positive rNPV) = 62%.
          </p>
          <apx-chart [series]="rnpvChart.series" [chart]="rnpvChart.chart"
            [colors]="rnpvChart.colors" [xaxis]="rnpvChart.xaxis"
            [yaxis]="rnpvChart.yaxis" [plotOptions]="rnpvChart.plotOptions"
            [annotations]="rnpvChart.annotations" [tooltip]="rnpvChart.tooltip"
            [grid]="rnpvChart.grid" [theme]="rnpvChart.theme">
          </apx-chart>
          <div class="metric-row">
            <div class="metric"><span class="metric__label">P5</span><span class="metric__value text-red">−$313M</span></div>
            <div class="metric"><span class="metric__label">Mean</span><span class="metric__value text-blue">$142M</span></div>
            <div class="metric"><span class="metric__label">P95</span><span class="metric__value text-green">$521M</span></div>
            <div class="metric"><span class="metric__label">Prob +rNPV</span><span class="metric__value text-green">62%</span></div>
          </div>
        </div>
        <div class="card">
          <div class="card__header">
            <span class="card__title">Patent Cliff Revenue Erosion</span>
            <span class="badge green">Biologic $580M premium</span>
          </div>
          <apx-chart [series]="cliffChart.series" [chart]="cliffChart.chart"
            [stroke]="cliffChart.stroke" [colors]="cliffChart.colors"
            [fill]="cliffChart.fill" [xaxis]="cliffChart.xaxis"
            [yaxis]="cliffChart.yaxis" [annotations]="cliffChart.annotations"
            [tooltip]="cliffChart.tooltip" [legend]="cliffChart.legend"
            [grid]="cliffChart.grid" [theme]="cliffChart.theme">
          </apx-chart>
        </div>
      </div>

      <div class="card">
        <div class="card__header">
          <span class="card__title">Pipeline Monitor — Trial Enrollment vs Target</span>
          <span class="badge orange">2 At-Risk</span>
        </div>
        <apx-chart [series]="pipelineChart.series" [chart]="pipelineChart.chart"
          [colors]="pipelineChart.colors" [xaxis]="pipelineChart.xaxis"
          [yaxis]="pipelineChart.yaxis" [plotOptions]="pipelineChart.plotOptions"
          [annotations]="pipelineChart.annotations" [legend]="pipelineChart.legend"
          [tooltip]="pipelineChart.tooltip" [grid]="pipelineChart.grid"
          [theme]="pipelineChart.theme">
        </apx-chart>
        <div class="table-responsive-wrap">
          <table class="data-table" style="margin-top:12px">
            <thead><tr><th>Asset</th><th>Indication</th><th>Phase</th><th>Enrollment</th><th>rNPV</th><th>Status</th></tr></thead>
            <tbody>
              <tr *ngFor="let p of pharmaAssets">
                <td class="text-blue font-bold">{{ p.asset }}</td>
                <td>{{ p.indication }}</td>
                <td>{{ p.phase }}</td>
                <td>
                  <div class="flex items-center gap-8">
                    <div class="progress-bar" style="width:80px">
                      <div class="progress-bar__fill"
                        [style.width]="p.enrollment+'%'"
                        [style.background]="p.enrollment>=80?'var(--green)':p.enrollment>=60?'var(--yellow)':'var(--red)'">
                      </div>
                    </div>
                    <span>{{ p.enrollment }}%</span>
                  </div>
                </td>
                <td class="text-green font-bold">\${{ p.rnpv }}M</td>
                <td><span class="badge" [ngClass]="pharmaColor(p.status)">{{ p.status }}</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </ng-container>
  `,
  styles: [`
    .inner-tabs {
      display: flex;
      gap: 4px;
      margin-bottom: 20px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0;
    }
    .inner-tab {
      padding: 8px 18px;
      font-size: 13px;
      font-weight: 500;
      color: var(--text-secondary);
      background: none;
      border: none;
      border-bottom: 2px solid transparent;
      cursor: pointer;
      transition: color 0.15s, border-color 0.15s;
      margin-bottom: -1px;
    }
    .inner-tab:hover { color: var(--text-primary); }
    .inner-tab--active {
      color: var(--blue);
      border-bottom-color: var(--blue);
    }
    @media (max-width: 480px) {
      .inner-tab { padding: 8px 12px; font-size: 12px; }
    }
  `]
})
export class RiskComponent implements OnInit {
  private svc = inject(MockDataService);
  tab = signal<'credit' | 'pharma'>('credit');

  creditKpis: KpiCard[] = [];
  pharmaKpis: KpiCard[] = [];
  hospitals: any[] = [];

  pharmaAssets = [
    { asset:'PH-01', indication:'Oncology',   phase:'Phase III', enrollment:88, rnpv:142, status:'on-track' },
    { asset:'PH-02', indication:'Cardiology', phase:'Phase III', enrollment:73, rnpv: 98, status:'at-risk'  },
    { asset:'PH-03', indication:'Oncology',   phase:'Phase III', enrollment:91, rnpv:186, status:'on-track' },
    { asset:'PH-04', indication:'CNS',        phase:'Phase II',  enrollment:65, rnpv: 54, status:'at-risk'  },
    { asset:'PH-05', indication:'Autoimmune', phase:'Phase II',  enrollment:82, rnpv: 71, status:'on-track' },
    { asset:'PH-06', indication:'Rare',       phase:'Phase I',   enrollment:95, rnpv: 28, status:'on-track' },
    { asset:'PH-07', indication:'Oncology',   phase:'Phase I',   enrollment:44, rnpv: 19, status:'safety-flag'},
  ];

  rocChart: any = {};
  spreadChart: any = {};
  scoreDistChart: any = {};
  rnpvChart: any = {};
  cliffChart: any = {};
  pipelineChart: any = {};

  ngOnInit(): void {
    this.creditKpis = this.svc.creditRiskKpis();
    this.pharmaKpis = this.svc.pharmaKpis();
    this.hospitals   = this.svc.hospitalScorecard();
    this.buildCreditCharts();
    this.buildPharmaCharts();
  }

  private buildCreditCharts(): void {
    const pts = (auroc: number) => Array.from({length:51}, (_,i) => {
      const fpr = i/50;
      return parseFloat(Math.min(1, fpr + (auroc-0.5)*2*Math.sqrt(fpr*(1-fpr)+0.001)).toFixed(3));
    });
    const fprs = Array.from({length:51},(_,i) => parseFloat((i/50).toFixed(2)));
    this.rocChart = {
      series: [
        { name:'HealthRisk AI (0.851)', data: fprs.map((x,i)=>({x,y:pts(0.851)[i]})) },
        { name:'Financial Only (0.742)', data: fprs.map((x,i)=>({x,y:pts(0.742)[i]})) },
        { name:'Diagonal', data: fprs.map(x=>({x,y:x})) },
      ],
      chart:  { type:'line', height:260, background:'transparent', toolbar:{show:false} },
      stroke: { curve:'straight', width:[3,2,1], dashArray:[0,0,4] },
      colors: ['#34a853','#f77f00','#30363d'],
      xaxis:  { type:'numeric', title:{text:'FPR',style:{color:'#8b949e'}}, labels:{style:{colors:'#8b949e'}}, min:0, max:1 },
      yaxis:  { min:0, max:1, title:{text:'TPR',style:{color:'#8b949e'}}, labels:{style:{colors:'#8b949e'}} },
      tooltip:{ theme:'dark' },
      legend: { labels:{colors:'#8b949e'} },
      grid:   { borderColor:'#30363d', strokeDashArray:3 },
      theme:  { mode:'dark' },
    };

    const curves = this.svc.bondSpreadHistory();
    this.spreadChart = {
      series: [
        { name:'Portfolio Avg (bps)', data: curves[0].map((p:any)=>p.y) },
        { name:'Distressed (Oakridge)', data: curves[1].map((p:any)=>p.y) },
      ],
      chart:  { type:'area', height:260, background:'transparent', toolbar:{show:false} },
      stroke: { curve:'smooth', width:[2,2] },
      colors: ['#1a73e8','#ea4335'],
      fill:   { type:'gradient', gradient:{opacityFrom:0.15,opacityTo:0.02} },
      xaxis:  { categories: curves[0].map((p:any)=>p.x), labels:{style:{colors:'#8b949e'}} },
      yaxis:  { labels:{formatter:(v:number)=>`${v} bps`,style:{colors:'#8b949e'}} },
      tooltip:{ theme:'dark', y:{formatter:(v:number)=>`${v} bps`} },
      legend: { labels:{colors:'#8b949e'} },
      grid:   { borderColor:'#30363d', strokeDashArray:3 },
      theme:  { mode:'dark' },
    };

    const cats = Array.from({length:20},(_,i)=>(i*5).toString());
    this.scoreDistChart = {
      series: [
        { name:'Stable (PD < 3%)',  data:[0,0,1,3,5,8,12,15,18,16,12,8,5,3,2,1,0,0,0,0] },
        { name:'Distressed (PD > 3%)', data:[0,0,0,1,3,6,10,14,12,10,8,6,4,2,1,0,0,0,0,0] },
      ],
      chart:  { type:'bar', height:220, background:'transparent', toolbar:{show:false} },
      colors: ['#34a853','#ea4335'],
      xaxis:  { categories:cats, title:{text:'Credit Score',style:{color:'#8b949e'}}, labels:{style:{colors:'#8b949e',fontSize:'9px'}} },
      yaxis:  { labels:{style:{colors:'#8b949e'}} },
      plotOptions: { bar:{borderRadius:2} },
      tooltip:{ theme:'dark' },
      legend: { labels:{colors:'#8b949e'} },
      grid:   { borderColor:'#30363d' },
      theme:  { mode:'dark' },
    };
  }

  private buildPharmaCharts(): void {
    const bins = Array.from({length:30},(_,i)=>-350+i*30);
    const dist  = [0,0,1,2,4,7,11,16,20,22,18,14,10,7,5,4,3,2,2,1,1,1,0,0,0,0,0,0,0,0];
    this.rnpvChart = {
      series: [{ name:'Scenarios', data:dist }],
      chart:  { type:'bar', height:250, background:'transparent', toolbar:{show:false} },
      colors: ['#1a73e8'],
      xaxis:  { categories:bins.map(b=>`$${b}M`), labels:{rotate:-45,style:{colors:'#8b949e',fontSize:'9px'}} },
      yaxis:  { labels:{style:{colors:'#8b949e'}} },
      plotOptions: { bar:{borderRadius:2,columnWidth:'90%'} },
      annotations:{ xaxis:[{x:'$140M',borderColor:'#34a853',label:{text:'Mean rNPV $142M',style:{color:'#34a853',background:'transparent'}}}] },
      tooltip:{ theme:'dark', y:{formatter:(v:number)=>`${v} scenarios`} },
      grid:   { borderColor:'#30363d', strokeDashArray:3 },
      theme:  { mode:'dark' },
    };

    const years = ['Y1','Y2','Y3','Y4','Y5','Y6','Y7','Y8','Y9','Y10'];
    this.cliffChart = {
      series: [
        { name:'Small Molecule', data:[100,100,100,52,28,18,12,10,9,8] },
        { name:'Biologic',       data:[100,100,100,90,78,65,55,46,40,36] },
      ],
      chart:  { type:'area', height:250, background:'transparent', toolbar:{show:false} },
      stroke: { curve:'smooth', width:[2,2] },
      colors: ['#ea4335','#34a853'],
      fill:   { type:'gradient', gradient:{opacityFrom:0.25,opacityTo:0.02} },
      xaxis:  { categories:years, labels:{style:{colors:'#8b949e'}} },
      yaxis:  { max:110, labels:{formatter:(v:number)=>`${v}%`,style:{colors:'#8b949e'}} },
      annotations:{ yaxis:[{y:50,borderColor:'#8b949e',label:{text:'50% threshold',style:{color:'#8b949e',background:'transparent'}}}] },
      tooltip:{ theme:'dark', y:{formatter:(v:number)=>`${v}% of peak`} },
      legend: { labels:{colors:'#8b949e'} },
      grid:   { borderColor:'#30363d', strokeDashArray:3 },
      theme:  { mode:'dark' },
    };

    this.pipelineChart = {
      series: [{ name:'Enrollment %', data:this.pharmaAssets.map(p=>p.enrollment) }],
      chart:  { type:'bar', height:220, background:'transparent', toolbar:{show:false} },
      colors: this.pharmaAssets.map(p=>p.status==='on-track'?'#34a853':p.status==='at-risk'?'#f77f00':'#ea4335'),
      xaxis:  { categories:this.pharmaAssets.map(p=>p.asset), labels:{style:{colors:'#8b949e'}} },
      yaxis:  { max:100, labels:{formatter:(v:number)=>`${v}%`,style:{colors:'#8b949e'}} },
      plotOptions: { bar:{borderRadius:4,distributed:true} },
      annotations:{ yaxis:[{y:80,borderColor:'#f9c74f',label:{text:'80% minimum',style:{color:'#f9c74f',background:'transparent',fontSize:'10px'}}}] },
      legend: { show:false },
      tooltip:{ theme:'dark', y:{formatter:(v:number)=>`${v}% enrolled`} },
      grid:   { borderColor:'#30363d', strokeDashArray:3 },
      theme:  { mode:'dark' },
    };
  }

  creditColor(s: string): string { return s==='stable'?'green':s==='watch'?'yellow':'red'; }
  pharmaColor(s: string): string { return s==='on-track'?'green':s==='at-risk'?'orange':'red'; }
}
