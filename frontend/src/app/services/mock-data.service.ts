import { Injectable } from '@angular/core';

// ── Types ─────────────────────────────────────────────────────────────────────
export interface KpiCard {
  id: string;
  label: string;
  value: string;
  rawValue: number;
  unit: string;
  delta: string;
  deltaDir: 'pos' | 'neg' | 'neu';
  icon: string;
  color: string;
  description: string;
}

export interface ChartSeries { name: string; data: number[]; }
export interface TimePoint   { x: string; y: number; }

// ── Service ──────────────────────────────────────────────────────────────────
@Injectable({ providedIn: 'root' })
export class MockDataService {

  // ── Portfolio overview KPIs (12) ─────────────────────────────────────────
  portfolioKpis(): KpiCard[] {
    return [
      { id:'aum',       label:'Portfolio AUM',        value:'$500M',   rawValue:500,  unit:'M',   delta:'+4.2%', deltaDir:'pos', icon:'💼', color:'#1a73e8', description:'Total assets under management' },
      { id:'return',    label:'YTD Return',            value:'12.2%',   rawValue:12.2, unit:'%',   delta:'+2.1pp vs benchmark', deltaDir:'pos', icon:'📈', color:'#34a853', description:'Year-to-date portfolio return' },
      { id:'sharpe',    label:'Sharpe Ratio',          value:'1.42',    rawValue:1.42, unit:'',    delta:'+0.18 vs prior yr', deltaDir:'pos', icon:'⚖️', color:'#0ea5e9', description:'Risk-adjusted return ratio' },
      { id:'maxdd',     label:'Max Drawdown',          value:'-8.4%',   rawValue:-8.4, unit:'%',   delta:'+2.1pp improvement', deltaDir:'pos', icon:'📉', color:'#ea4335', description:'Maximum peak-to-trough loss' },
      { id:'var95',     label:'VaR 95% (1-day)',       value:'$3.2M',   rawValue:3.2,  unit:'M',   delta:'-$0.4M vs limit', deltaDir:'pos', icon:'🎯', color:'#f9c74f', description:'Value at risk at 95% confidence' },
      { id:'cvar',      label:'CVaR 99%',              value:'$8.7M',   rawValue:8.7,  unit:'M',   delta:'Within limit', deltaDir:'neu', icon:'🔐', color:'#a855f7', description:'Conditional value at risk' },
      { id:'vol',       label:'Annualised Volatility', value:'14.3%',   rawValue:14.3, unit:'%',   delta:'-1.2pp vs Q3', deltaDir:'pos', icon:'〰️', color:'#f77f00', description:'Portfolio return volatility' },
      { id:'beta',      label:'Market Beta',           value:'0.61',    rawValue:0.61, unit:'',    delta:'Low correlation', deltaDir:'pos', icon:'📊', color:'#0ea5e9', description:'Sensitivity to market movements' },
      { id:'risksc',    label:'Aggregate Risk Score',  value:'62/100',  rawValue:62,   unit:'',    delta:'-5 pts (improved)', deltaDir:'pos', icon:'🛡️', color:'#34a853', description:'Composite portfolio risk score' },
      { id:'alerts',    label:'Active Alerts',         value:'3',       rawValue:3,    unit:'',    delta:'1 critical', deltaDir:'neg', icon:'🚨', color:'#ea4335', description:'Open risk alerts requiring action' },
      { id:'horizon',   label:'Simulation Horizon',    value:'10 yrs',  rawValue:10,   unit:'yrs', delta:'40 quarters', deltaDir:'neu', icon:'🕐', color:'#8b949e', description:'Monte Carlo time horizon' },
      { id:'conf',      label:'Model Confidence',      value:'87%',     rawValue:87,   unit:'%',   delta:'Ensemble avg.', deltaDir:'pos', icon:'🤖', color:'#1a73e8', description:'Average model prediction confidence' },
    ];
  }

  // ── Insurance KPIs (12) ──────────────────────────────────────────────────
  insuranceKpis(): KpiCard[] {
    return [
      { id:'plr',    label:'Pure Loss Ratio',       value:'71.3%',  rawValue:71.3,  unit:'%',  delta:'-2.1pp YoY', deltaDir:'pos', icon:'📋', color:'#1a73e8', description:'Claims / Net premiums earned' },
      { id:'mlr',    label:'Medical Loss Ratio',    value:'84.2%',  rawValue:84.2,  unit:'%',  delta:'ACA compliant ≥80%', deltaDir:'pos', icon:'🏥', color:'#34a853', description:'Medical spend / premium revenue' },
      { id:'pmpm',   label:'PMPM Spend',            value:'$892',   rawValue:892,   unit:'$',  delta:'+3.1% vs prior', deltaDir:'neg', icon:'💊', color:'#f77f00', description:'Per-member-per-month cost' },
      { id:'ibnr',   label:'IBNR Reserve',          value:'$24.1M', rawValue:24.1,  unit:'M$', delta:'+$1.2M this qtr', deltaDir:'neg', icon:'📦', color:'#a855f7', description:'Incurred-but-not-reported reserve' },
      { id:'ibnrrat',label:'IBNR Ratio',            value:'4.8%',   rawValue:4.8,   unit:'%',  delta:'Target ≤5% ✓', deltaDir:'pos', icon:'✅', color:'#34a853', description:'IBNR / Total earned premium' },
      { id:'pred',   label:'Predictive Ratio',      value:'0.99',   rawValue:0.99,  unit:'',   delta:'Target 0.95–1.05 ✓', deltaDir:'pos', icon:'🎯', color:'#34a853', description:'Predicted / actual claims ratio' },
      { id:'r2',     label:'Cost Model R²',         value:'0.28',   rawValue:0.28,  unit:'',   delta:'+115% vs CMS-HCC', deltaDir:'pos', icon:'📐', color:'#0ea5e9', description:'Cost prediction variance explained' },
      { id:'mape',   label:'Cost Model MAPE',       value:'52%',    rawValue:52,    unit:'%',  delta:'-16pp vs baseline', deltaDir:'pos', icon:'📏', color:'#34a853', description:'Mean absolute % prediction error' },
      { id:'mems',   label:'Enrolled Members',      value:'50,000', rawValue:50000, unit:'',   delta:'+1,200 this qtr', deltaDir:'pos', icon:'👥', color:'#1a73e8', description:'Active enrolled membership' },
      { id:'highr',  label:'High-Risk Members',     value:'8,420',  rawValue:8420,  unit:'',   delta:'16.8% of book', deltaDir:'neu', icon:'⚠️', color:'#f9c74f', description:'Members in top 20% risk tier' },
      { id:'prem',   label:'Avg Premium PMPM',      value:'$1,062', rawValue:1062,  unit:'$',  delta:'+2.3% renewal', deltaDir:'pos', icon:'💰', color:'#34a853', description:'Average monthly premium per member' },
      { id:'ecr',    label:'Expected Claims Ratio', value:'78.6%',  rawValue:78.6,  unit:'%',  delta:'Model forecast', deltaDir:'neu', icon:'🔮', color:'#a855f7', description:'Model-projected loss ratio for next year' },
    ];
  }

  // ── Credit Risk KPIs (10) ────────────────────────────────────────────────
  creditRiskKpis(): KpiCard[] {
    return [
      { id:'auroc',  label:'PD Model AUROC',      value:'0.851',   rawValue:0.851, unit:'',   delta:'+0.109 vs fin-only', deltaDir:'pos', icon:'📊', color:'#34a853', description:'Area under ROC curve for default model' },
      { id:'gini',   label:'Gini Coefficient',    value:'0.702',   rawValue:0.702, unit:'',   delta:'Excellent (>0.50)', deltaDir:'pos', icon:'🎯', color:'#34a853', description:'Model discriminatory power' },
      { id:'ks',     label:'KS Statistic',         value:'0.421',   rawValue:0.421, unit:'',   delta:'Target >0.30 ✓', deltaDir:'pos', icon:'📐', color:'#34a853', description:'Kolmogorov-Smirnov separation' },
      { id:'avpd',   label:'Portfolio Avg PD',     value:'2.3%',    rawValue:2.3,   unit:'%',  delta:'+0.2pp vs last qtr', deltaDir:'neg', icon:'📉', color:'#f9c74f', description:'Avg probability of default, 15 issuers' },
      { id:'bonds',  label:'Bond Issuers',         value:'15',      rawValue:15,    unit:'',   delta:'2 on watchlist', deltaDir:'neg', icon:'🏦', color:'#f77f00', description:'Active hospital bond positions' },
      { id:'sprd',   label:'Avg Bond Spread',      value:'185 bps', rawValue:185,   unit:'bps',delta:'+12 bps this qtr', deltaDir:'neg', icon:'📈', color:'#ea4335', description:'Credit spread above Treasury benchmark' },
      { id:'exp',    label:'Total Exposure',       value:'$280M',   rawValue:280,   unit:'M$', delta:'56% of portfolio', deltaDir:'neu', icon:'💼', color:'#1a73e8', description:'Total hospital bond exposure' },
      { id:'ewalerts',label:'Early Warnings',      value:'4',       rawValue:4,     unit:'',   delta:'1 critical, 3 watch', deltaDir:'neg', icon:'🚨', color:'#ea4335', description:'Active hospital deterioration alerts' },
      { id:'recov',  label:'Avg Recovery Rate',    value:'68%',     rawValue:68,    unit:'%',  delta:'Hospital sector avg', deltaDir:'neu', icon:'♻️', color:'#0ea5e9', description:'Expected recovery in default scenario' },
      { id:'ecl',    label:'Expected Credit Loss', value:'$6.4M',   rawValue:6.4,   unit:'M$', delta:'1.28% of exposure', deltaDir:'neu', icon:'⚖️', color:'#a855f7', description:'IFRS 9 expected credit loss' },
    ];
  }

  // ── Pharma KPIs (10) ─────────────────────────────────────────────────────
  pharmaKpis(): KpiCard[] {
    return [
      { id:'pipes',  label:'Pipeline Assets',       value:'20',     rawValue:20,    unit:'',   delta:'3 Phase III', deltaDir:'pos', icon:'🧬', color:'#1a73e8', description:'Active pharma/biotech equity positions' },
      { id:'rnpv',   label:'Portfolio rNPV',        value:'$142M',  rawValue:142,   unit:'M$', delta:'+$18M this qtr', deltaDir:'pos', icon:'💊', color:'#34a853', description:'Risk-adjusted net present value' },
      { id:'p3succ', label:'Phase III Success Rate',value:'48%',    rawValue:48,    unit:'%',  delta:'Oncology adjusted', deltaDir:'neu', icon:'🔬', color:'#0ea5e9', description:'ClinicalTrials.gov aggregate rate' },
      { id:'p3val',  label:'Phase III Exposure',    value:'$210M',  rawValue:210,   unit:'M$', delta:'3 active trials', deltaDir:'neu', icon:'⚗️', color:'#f77f00', description:'Total exposure in Phase III assets' },
      { id:'cliff',  label:'Patent Cliffs (2yr)',   value:'$340M',  rawValue:340,   unit:'M$', delta:'2 assets expiring', deltaDir:'neg', icon:'⛰️', color:'#ea4335', description:'Revenue at risk from upcoming patent expiry' },
      { id:'biosim', label:'Biologic Premium',      value:'$580M',  rawValue:580,   unit:'M$', delta:'vs small molecule', deltaDir:'pos', icon:'🧪', color:'#a855f7', description:'10-yr cumulative biosimilar delay premium' },
      { id:'faers',  label:'FAERS Safety Signals',  value:'2',      rawValue:2,     unit:'',   delta:'Review triggered', deltaDir:'neg', icon:'⚠️', color:'#f9c74f', description:'Active disproportionality signals' },
      { id:'irr',    label:'Portfolio IRR',         value:'14.2%',  rawValue:14.2,  unit:'%',  delta:'+2.3pp vs WACC', deltaDir:'pos', icon:'📈', color:'#34a853', description:'Internal rate of return, pharma sleeve' },
      { id:'enroll', label:'Trial Enrollment Rate', value:'73%',    rawValue:73,    unit:'%',  delta:'-4pp vs plan', deltaDir:'neg', icon:'👤', color:'#f77f00', description:'Avg enrollment vs target across Phase II/III' },
      { id:'tob',    label:'Time-to-Opportunity',   value:'3.2 yrs',rawValue:3.2,   unit:'yrs',delta:'Phase III → launch', deltaDir:'neu', icon:'⏱️', color:'#0ea5e9', description:'Expected years to market launch' },
    ];
  }

  // ── Chart data ────────────────────────────────────────────────────────────

  portfolioEquityCurve(): { x: string; y: number }[][] {
    const quarters = Array.from({length: 41}, (_, i) => `Q${i === 0 ? '0' : i}`);
    const aiCurve   = [500];
    const plrCurve  = [500];
    const bench     = [500];
    let ai = 500, pl = 500, bk = 500;
    for (let i = 1; i <= 40; i++) {
      const isPandemic = i === 8;
      ai += ai * (isPandemic ? -0.12 : 0.038 + (Math.random() - 0.45) * 0.02);
      pl += pl * (isPandemic ? -0.18 : 0.028 + (Math.random() - 0.48) * 0.02);
      bk += bk * 0.034;
      aiCurve.push(Math.round(ai * 10) / 10);
      plrCurve.push(Math.round(pl * 10) / 10);
      bench.push(Math.round(bk * 10) / 10);
    }
    return [
      quarters.map((x, i) => ({ x, y: aiCurve[i] })),
      quarters.map((x, i) => ({ x, y: plrCurve[i] })),
      quarters.map((x, i) => ({ x, y: bench[i] })),
    ];
  }

  rocCurves(): { name: string; data: { x: number; y: number }[] }[] {
    const pts = (auroc: number) => {
      const n = 50;
      return Array.from({length: n + 1}, (_, i) => {
        const fpr = i / n;
        const tpr = Math.min(1, fpr + (auroc - 0.5) * 2 * Math.sqrt(fpr * (1 - fpr) + 0.001));
        return { x: Math.round(fpr * 1000) / 1000, y: Math.round(tpr * 1000) / 1000 };
      });
    };
    return [
      { name: 'Stacking Ensemble (0.831)', data: pts(0.831) },
      { name: 'XGBoost (0.812)',           data: pts(0.812) },
      { name: 'LightGBM (0.794)',          data: pts(0.794) },
      { name: 'ClinicalBERT (0.783)',      data: pts(0.783) },
      { name: 'Random Baseline (0.500)',   data: pts(0.500) },
    ];
  }

  ibnrTriangle(): number[][] {
    return [
      [18.2, 21.4, 22.8, 23.5, 23.9],
      [19.1, 22.6, 24.1, 24.8,    0],
      [20.3, 23.8, 25.4,    0,    0],
      [21.0, 24.7,    0,    0,    0],
      [22.4,    0,    0,    0,    0],
    ];
  }

  riskStratification(): { tier: string; count: number; pmpm: number; ratio: number }[] {
    return [
      { tier: 'Very High (>0.7)',  count: 1820,  pmpm: 4210, ratio: 4.72 },
      { tier: 'High (0.5–0.7)',    count: 6600,  pmpm: 2180, ratio: 2.44 },
      { tier: 'Medium (0.3–0.5)',  count: 14200, pmpm: 980,  ratio: 1.10 },
      { tier: 'Low (<0.3)',        count: 27380, pmpm: 410,  ratio: 0.46 },
    ];
  }

  hospitalScorecard(): { name: string; score: number; pd: number; spread: number; status: string }[] {
    return [
      { name: 'Metro General',     score: 72, pd: 1.2, spread: 120, status: 'stable' },
      { name: 'Riverside Medical', score: 68, pd: 1.8, spread: 145, status: 'stable' },
      { name: 'Valley Health',     score: 55, pd: 3.1, spread: 198, status: 'watch' },
      { name: 'Summit Hospital',   score: 51, pd: 4.2, spread: 237, status: 'watch' },
      { name: 'Oakridge Medical',  score: 44, pd: 6.8, spread: 312, status: 'critical' },
      { name: 'Pine Valley HC',    score: 79, pd: 0.8, spread: 98,  status: 'stable' },
      { name: 'Coastal Medical',   score: 63, pd: 2.3, spread: 165, status: 'stable' },
      { name: 'Northside Health',  score: 48, pd: 5.1, spread: 278, status: 'watch' },
    ];
  }

  shapValues(): { feature: string; value: number }[] {
    return [
      { feature: 'Prior Admissions',     value: 0.142 },
      { feature: 'HCC Risk Score',        value: 0.118 },
      { feature: 'Age',                   value: 0.097 },
      { feature: 'ER Visits (12m)',        value: 0.086 },
      { feature: 'Chronic Conditions',    value: 0.074 },
      { feature: 'LOS Last Admission',    value: 0.061 },
      { feature: 'Comorbidity Index',     value: 0.055 },
      { feature: 'HbA1c Level',           value: 0.044 },
      { feature: 'Creatinine Slope',      value: 0.038 },
      { feature: 'Medication Adherence',  value: -0.029 },
      { feature: 'BMI',                   value: 0.027 },
      { feature: 'No Prior Surgery',      value: -0.019 },
    ];
  }

  pipelineStatus(): { id: string; name: string; status: string; progress: number; lastRun: string; duration: string }[] {
    return [
      { id:'dp1', name:'MIMIC-IV Data Ingestion',    status:'success', progress:100, lastRun:'2026-07-09 14:02', duration:'4m 23s' },
      { id:'dp2', name:'WHO GHO Fetch',              status:'success', progress:100, lastRun:'2026-07-09 14:06', duration:'1m 12s' },
      { id:'dp3', name:'FDA FAERS Update',           status:'running', progress:67,  lastRun:'2026-07-09 17:30', duration:'ongoing' },
      { id:'dp4', name:'ClinicalTrials.gov Sync',    status:'success', progress:100, lastRun:'2026-07-09 08:00', duration:'6m 41s' },
      { id:'dp5', name:'Feature Engineering',        status:'success', progress:100, lastRun:'2026-07-09 14:15', duration:'2m 55s' },
      { id:'ml1', name:'Readmission Model Retrain',  status:'queued',  progress:0,   lastRun:'2026-07-08 22:00', duration:'—' },
      { id:'ml2', name:'Cost Predictor Update',      status:'queued',  progress:0,   lastRun:'2026-07-08 22:00', duration:'—' },
      { id:'ml3', name:'Hospital PD Model',          status:'success', progress:100, lastRun:'2026-07-09 00:05', duration:'8m 14s' },
      { id:'ml4', name:'Ensemble Stacking',          status:'success', progress:100, lastRun:'2026-07-09 00:14', duration:'3m 08s' },
      { id:'ml5', name:'SHAP Explainability',        status:'failed',  progress:45,  lastRun:'2026-07-09 01:00', duration:'failed at 45%' },
      { id:'sm1', name:'Monte Carlo Simulation',     status:'success', progress:100, lastRun:'2026-07-09 06:00', duration:'12m 30s' },
      { id:'sm2', name:'Scenario Stress Tests',      status:'success', progress:100, lastRun:'2026-07-09 06:12', duration:'5m 02s' },
    ];
  }

  simulationState() {
    return {
      quarter: 8,
      portfolioValue: 514.3,
      aiValue: 543.2,
      benchValue: 518.8,
      score: { total: 612, performance: 248, risk: 201, clinical: 120, speed: 43 },
      aiScore: { total: 875, performance: 368, risk: 251, clinical: 174, speed: 82 },
      currentScenario: 'Pandemic Outbreak',
      severity: 'Severe',
      activeAlerts: [
        { type: 'Epidemic', message: 'R₀ = 2.4 detected in WHO surveillance data', severity: 'critical' },
        { type: 'Credit',   message: 'Oakridge Medical: CMI declining −0.3 for 3 consecutive quarters', severity: 'warning' },
        { type: 'Pharma',   message: 'FAERS signal: Disproportionality ROR > 3.0 for Asset PH-07', severity: 'warning' },
      ],
    };
  }

  aiInsights(): { id: string; title: string; category: string; priority: string; summary: string; timestamp: string; confidence: number }[] {
    return [
      {
        id: 'ins1',
        title: 'Pandemic Early Warning — R₀ Exceeds Threshold',
        category: 'Epidemiology',
        priority: 'critical',
        summary: 'WHO GHO surveillance data shows R₀ = 2.4 in 3 WHO regions simultaneously. Historical patterns suggest 4–6 week window before financial market reaction. Recommend: increase IBNR reserves by 15%, reduce hospital bond duration, review ICU capacity utilisation.',
        timestamp: '2026-07-09 17:15',
        confidence: 91,
      },
      {
        id: 'ins2',
        title: 'Oakridge Medical — Downgrade Alert',
        category: 'Credit Risk',
        priority: 'high',
        summary: 'Case Mix Index has declined for 3 consecutive quarters (−0.31 cumulative). 30-day readmission rate increased from 14.2% to 17.8%. HCAHPS score fell below 3.5 stars. Model probability of default has risen from 4.2% to 6.8% (+62%). Implied spread widening: +45 bps.',
        timestamp: '2026-07-09 16:42',
        confidence: 84,
      },
      {
        id: 'ins3',
        title: 'Phase III Oncology Asset — Enrollment Shortfall',
        category: 'Pharma',
        priority: 'medium',
        summary: 'Trial enrollment at 73% of target at 18-month mark. Historical data shows <80% enrollment at this stage correlates with 23% higher trial failure rate. rNPV impact: −$14M. Consider reducing position by 30%.',
        timestamp: '2026-07-09 14:30',
        confidence: 76,
      },
      {
        id: 'ins4',
        title: 'Insurance Book — HbA1c Trend Alert',
        category: 'Insurance',
        priority: 'medium',
        summary: '2,840 members showing HbA1c slope > +0.3 per measurement over last 12 months. Model projects +$36,510 average annual cost increase per affected member. Portfolio impact: +$103M claims over 24 months if untreated. Recommend targeted diabetes management programme.',
        timestamp: '2026-07-09 12:00',
        confidence: 88,
      },
      {
        id: 'ins5',
        title: 'Patent Cliff — $340M Revenue at Risk',
        category: 'Pharma',
        priority: 'high',
        summary: 'Two portfolio assets face patent expiry within 24 months. Combined peak-year revenue: $340M. Small molecule erosion modelled at 50% by month 30. Recommend rebalancing toward biologic assets with longer biosimilar runway.',
        timestamp: '2026-07-09 09:00',
        confidence: 95,
      },
      {
        id: 'ins6',
        title: 'IBNR Development — Adverse Emergence',
        category: 'Insurance',
        priority: 'low',
        summary: 'Q2 development triangle shows ATA factor 1.08 vs expected 1.04. Prior period development adverse by $1.2M. Reserve strengthening of $800K recommended for Q3 close.',
        timestamp: '2026-07-08 18:00',
        confidence: 79,
      },
    ];
  }

  epidemicData(): { country: string; r0: number; growth: number; alert: boolean }[] {
    return [
      { country: 'USA',   r0: 1.2, growth: 8.4,  alert: false },
      { country: 'China', r0: 2.4, growth: 34.2, alert: true  },
      { country: 'EU',    r0: 1.8, growth: 15.6, alert: true  },
      { country: 'India', r0: 1.4, growth: 11.1, alert: false },
      { country: 'Brazil',r0: 1.6, growth: 13.0, alert: false },
    ];
  }

  pdpAgeData(): { x: number; y: number }[] {
    return Array.from({length: 60}, (_, i) => {
      const age = 20 + i;
      const risk = age < 45 ? 0.12 + age * 0.001
        : age < 65 ? 0.15 + (age - 45) * 0.004
        : 0.23 + (age - 65) * 0.012;
      return { x: age, y: Math.min(0.9, Math.round(risk * 1000) / 1000) };
    });
  }

  pdpHba1cData(): { x: number; y: number }[] {
    return Array.from({length: 50}, (_, i) => {
      const hba = 4.0 + i * 0.18;
      const risk = hba < 7.0 ? 0.12 + hba * 0.01
        : 0.12 + 7.0 * 0.01 + Math.pow(hba - 7.0, 1.6) * 0.06;
      return { x: Math.round(hba * 10) / 10, y: Math.min(0.95, Math.round(risk * 1000) / 1000) };
    });
  }

  bondSpreadHistory(): { x: string; y: number }[][] {
    const months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
    const baseline = [140, 142, 138, 145, 148, 155, 162, 170, 175, 178, 182, 185];
    const distressed = [180, 185, 190, 200, 215, 230, 248, 265, 278, 289, 300, 312];
    return [
      months.map((x, i) => ({ x, y: baseline[i] })),
      months.map((x, i) => ({ x, y: distressed[i] })),
    ];
  }
}
