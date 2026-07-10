import { Component, OnInit, inject, signal } from '@angular/core';
import { NgFor, NgClass, NgIf, DecimalPipe, SlicePipe } from '@angular/common';
import { NgApexchartsModule } from 'ng-apexcharts';
import { MockDataService } from '../../services/mock-data.service';
import { ApiService, FaersResponse, TrialsResponse } from '../../services/api.service';

@Component({
  selector: 'app-insights',
  standalone: true,
  imports: [NgFor, NgClass, NgIf, NgApexchartsModule, DecimalPipe, SlicePipe],
  templateUrl: './insights.component.html',
  styleUrl: './insights.component.scss',
})
export class InsightsComponent implements OnInit {
  private svc = inject(MockDataService);
  private api = inject(ApiService);
  Math = Math;

  tab = signal<'briefings' | 'explainability' | 'pipeline'>('briefings');

  // ── Briefings ──────────────────────────────────────────────────────────────
  insights = this.svc.aiInsights();
  selected = this.insights[0];

  // Live data from real APIs
  faersData = signal<FaersResponse | null>(null);
  trialsData = signal<TrialsResponse | null>(null);
  liveLoading = signal(true);

  modelCards = [
    { name:'Readmission Ensemble', auroc:0.831, r2:null, cindex:null, data:'MIMIC-IV', trained:'2026-07-01', version:'v2.4.1' },
    { name:'Cost Predictor (LightGBM)', auroc:null, r2:0.28, cindex:null, data:'CMS Claims', trained:'2026-07-01', version:'v1.8.0' },
    { name:'Hospital PD Model', auroc:0.851, r2:null, cindex:null, data:'CMS Hospital Compare', trained:'2026-07-05', version:'v3.1.0' },
    { name:'Survival Ensemble (DeepHit)', auroc:null, r2:null, cindex:0.762, data:'MIMIC-IV Survival', trained:'2026-07-01', version:'v1.2.0' },
  ];

  priorityColor(p: string): string { return p==='critical'?'red':p==='high'?'orange':p==='medium'?'yellow':'blue'; }
  categoryIcon(c: string): string { return ({Epidemiology:'🦠','Credit Risk':'🏦',Pharma:'💊',Insurance:'🏥'} as any)[c] ?? '🤖'; }

  // ── Explainability ─────────────────────────────────────────────────────────
  shapValues = this.svc.shapValues();
  waterfallOptions: any = {};
  globalShapOptions: any = {};
  pdpAgeOptions: any = {};
  pdpHba1cOptions: any = {};
  patient       = { age:72, priorAdmissions:2, hba1c:8.6, creatinine:1.9, erVisits:3 };
  counterfactual= { age:72, priorAdmissions:1, hba1c:7.0, creatinine:1.4, erVisits:1 };

  // ── Pipeline ───────────────────────────────────────────────────────────────
  pipelines   = signal(this.svc.pipelineStatus());
  runningId   = signal<string|null>(null);
  logs        = signal<string[]>([
    '[14:02:31] MIMIC-IV Data Ingestion — COMPLETE (4m 23s)',
    '[14:06:44] WHO GHO Fetch — COMPLETE (1m 12s)',
    '[17:30:01] FDA FAERS Update — IN PROGRESS (step 3/5: deduplication)',
    '[17:34:21] SHAP Explainability — FAILED: Memory limit exceeded at 45%',
  ]);

  ngOnInit(): void {
    // Fetch real FDA FAERS signals for aspirin
    this.api.faersSignals('aspirin').subscribe(d => { this.faersData.set(d); });
    // Fetch real ClinicalTrials.gov recruiting diabetes trials
    this.api.recruitingTrials('diabetes', 5).subscribe(d => {
      this.trialsData.set(d);
      this.liveLoading.set(false);
    });
    this.buildWaterfall();
    this.buildGlobalShap();
    this.buildPdpAge();
    this.buildPdpHba1c();
  }

  // ── Pipeline methods ───────────────────────────────────────────────────────
  pipelineStatusColor(s: string): string { return s==='success'?'green':s==='running'?'blue':s==='failed'?'red':s==='queued'?'yellow':'orange'; }
  pipelineStatusIcon(s: string): string  { return s==='success'?'✅':s==='running'?'⚙️':s==='failed'?'❌':s==='queued'?'⏳':'⚠️'; }

  runPipeline(id: string): void {
    this.runningId.set(id);
    this.logs.update(l => [...l, `[${new Date().toLocaleTimeString()}] Starting pipeline ${id}...`]);
    const pls = this.pipelines();
    const idx = pls.findIndex((p: any) => p.id === id);
    if (idx !== -1) {
      const updated = [...pls];
      updated[idx] = { ...updated[idx], status:'running', progress:0 };
      this.pipelines.set(updated);
    }
    let progress = 0;
    const interval = setInterval(() => {
      progress += Math.floor(Math.random()*15)+5;
      if (progress >= 100) {
        progress = 100;
        clearInterval(interval);
        const done = this.pipelines();
        const i = done.findIndex((p: any) => p.id === id);
        if (i !== -1) {
          const u = [...done];
          u[i] = { ...u[i], status:'success', progress:100, lastRun:new Date().toLocaleString(), duration:'~2m' };
          this.pipelines.set(u);
        }
        this.runningId.set(null);
        this.logs.update(l => [...l, `[${new Date().toLocaleTimeString()}] Pipeline ${id} — COMPLETE`]);
      } else {
        const mid = this.pipelines();
        const i = mid.findIndex((p: any) => p.id === id);
        if (i !== -1) {
          const u = [...mid]; u[i] = { ...u[i], progress };
          this.pipelines.set(u);
        }
        this.logs.update(l => [...l, `[${new Date().toLocaleTimeString()}] ${id}: ${progress}% complete`]);
      }
    }, 600);
  }

  // ── Explainability charts ──────────────────────────────────────────────────
  private buildWaterfall(): void {
    const c = [
      { label:'Prior Admissions', value:0.142 },
      { label:'HCC Score', value:0.091 },
      { label:'Age 72', value:0.071 },
      { label:'ER Visits (3)', value:0.058 },
      { label:'HbA1c 8.6%', value:0.044 },
      { label:'Medication Adherence', value:-0.038 },
      { label:'No Prior Surgery', value:-0.019 },
    ];
    this.waterfallOptions = {
      series: [{ name:'SHAP Value', data:c.map(x=>Math.abs(x.value)) }],
      chart:  { type:'bar', height:280, background:'transparent', toolbar:{show:false} },
      colors: c.map(x=>x.value>0?'#ea4335':'#34a853'),
      plotOptions: { bar:{horizontal:true,borderRadius:3,distributed:true} },
      xaxis:  { labels:{formatter:(v:number)=>v.toFixed(3),style:{colors:'#8b949e'}} },
      yaxis:  { categories:c.map(x=>x.label), labels:{style:{colors:'#8b949e',fontSize:'11px'}} },
      tooltip:{ theme:'dark', y:{formatter:(_v:number,opts:any)=>{ const x=c[opts.dataPointIndex]; return `${x.value>0?'+':''}${x.value.toFixed(3)}`; }} },
      legend: { show:false },
      grid:   { borderColor:'#30363d' },
      theme:  { mode:'dark' },
    };
  }

  private buildGlobalShap(): void {
    this.globalShapOptions = {
      series: [{ name:'Mean |SHAP|', data:this.shapValues.map((s:any)=>Math.abs(s.value)) }],
      chart:  { type:'bar', height:280, background:'transparent', toolbar:{show:false} },
      colors: ['#1a73e8'],
      plotOptions: { bar:{horizontal:true,borderRadius:3} },
      xaxis:  { labels:{formatter:(v:number)=>v.toFixed(3),style:{colors:'#8b949e'}} },
      yaxis:  { categories:this.shapValues.map((s:any)=>s.feature), labels:{style:{colors:'#8b949e',fontSize:'11px'}} },
      tooltip:{ theme:'dark', y:{formatter:(v:number)=>v.toFixed(3)} },
      grid:   { borderColor:'#30363d' },
      theme:  { mode:'dark' },
    };
  }

  private buildPdpAge(): void {
    const data = this.svc.pdpAgeData();
    this.pdpAgeOptions = {
      series: [{ name:'Readmission Risk', data:data.map((d:any)=>({x:d.x,y:d.y})) }],
      chart:  { type:'line', height:220, background:'transparent', toolbar:{show:false} },
      stroke: { curve:'smooth', width:2 },
      colors: ['#1a73e8'],
      xaxis:  { type:'numeric', title:{text:'Age (years)',style:{color:'#8b949e'}}, labels:{style:{colors:'#8b949e'}} },
      yaxis:  { max:0.9, labels:{formatter:(v:number)=>`${(v*100).toFixed(0)}%`,style:{colors:'#8b949e'}} },
      annotations: { xaxis:[{x:65,borderColor:'#f9c74f',label:{text:'Age 65',style:{color:'#f9c74f',background:'transparent'}}}] },
      tooltip:{ theme:'dark', y:{formatter:(v:number)=>`${(v*100).toFixed(1)}% risk`} },
      grid:   { borderColor:'#30363d', strokeDashArray:3 },
      theme:  { mode:'dark' },
    };
  }

  private buildPdpHba1c(): void {
    const data = this.svc.pdpHba1cData();
    this.pdpHba1cOptions = {
      series: [{ name:'Readmission Risk', data:data.map((d:any)=>({x:d.x,y:d.y})) }],
      chart:  { type:'line', height:220, background:'transparent', toolbar:{show:false} },
      stroke: { curve:'smooth', width:2 },
      colors: ['#a855f7'],
      xaxis:  { type:'numeric', title:{text:'HbA1c (%)',style:{color:'#8b949e'}}, labels:{style:{colors:'#8b949e'}} },
      yaxis:  { max:0.95, labels:{formatter:(v:number)=>`${(v*100).toFixed(0)}%`,style:{colors:'#8b949e'}} },
      annotations: { xaxis:[{x:7.0,borderColor:'#f9c74f',label:{text:'ADA Target 7.0%',style:{color:'#f9c74f',background:'transparent'}}}] },
      tooltip:{ theme:'dark', y:{formatter:(v:number)=>`${(v*100).toFixed(1)}% risk`} },
      grid:   { borderColor:'#30363d', strokeDashArray:3 },
      theme:  { mode:'dark' },
    };
  }
}
