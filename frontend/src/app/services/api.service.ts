import { Injectable, inject } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable, of, catchError, map, shareReplay } from 'rxjs';

// ── Backend URL ───────────────────────────────────────────────────────────────
// In production this is set via environment.ts / Vercel env vars.
// Falls back to localhost for local dev.
const API_BASE = (window as any).__env?.API_URL
  || (window.location.hostname === 'localhost' ? 'http://localhost:8000' : 'https://healthrisk-api-e5nv.onrender.com');

// ── Response types ─────────────────────────────────────────────────────────────

export interface LiveDashboard {
  who_life_expectancy: number;
  faers_aspirin_reactions: number;
  recruiting_trials: number;
  cms_hospitals_rated: number;
  data_freshness: { who: string; faers: string; trials: string };
  sources: string[];
  fallback?: boolean;
  error?: string;
}

export interface WhoIndicator {
  country: string;
  year: number;
  value: number;
}

export interface WhoResponse {
  indicator: string;
  values: WhoIndicator[];
  source: string;
  cached_at?: string;
  fallback?: boolean;
}

export interface FaersReaction {
  reaction: string;
  count: number;
}

export interface FaersResponse {
  drug: string;
  total_reports: number;
  top_reactions: FaersReaction[];
  source: string;
  cached_at?: string;
  fallback?: boolean;
}

export interface Trial {
  nct_id: string;
  title: string;
  phase: string;
  status: string;
  enrollment: number;
  start_date: string;
  locations_count: number;
}

export interface TrialsResponse {
  condition: string;
  total_found: number;
  trials: Trial[];
  source: string;
  cached_at?: string;
  fallback?: boolean;
}

export interface CmsHospital {
  provider_id: string;
  name: string;
  city: string;
  state: string;
  overall_rating: number | null;
  readmission_rate: number | null;
}

export interface CmsResponse {
  hospitals: CmsHospital[];
  source: string;
  count: number;
  cached_at?: string;
  fallback?: boolean;
}

export interface EdgarCompany {
  cik: string;
  name: string;
  ticker: string;
  sic: string;
  recent_10k_count: number;
}

export interface EdgarResponse {
  companies: EdgarCompany[];
  source: string;
  count: number;
  cached_at?: string;
  fallback?: boolean;
}

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
  uptime_seconds: number;
}

// ── Fallback mock data (used when API unreachable) ────────────────────────────

const MOCK_DASHBOARD: LiveDashboard = {
  who_life_expectancy: 73.4,
  faers_aspirin_reactions: 38026,
  recruiting_trials: 4872,
  cms_hospitals_rated: 3241,
  data_freshness: { who: 'mock', faers: 'mock', trials: 'mock' },
  sources: ['mock'],
  fallback: true,
};

const MOCK_WHO: WhoResponse = {
  indicator: 'life_expectancy',
  values: [
    { country: 'GLOBAL', year: 2019, value: 73.4 },
    { country: 'GLOBAL', year: 2018, value: 72.8 },
  ],
  source: 'mock',
  fallback: true,
};

const MOCK_FAERS: FaersResponse = {
  drug: 'aspirin',
  total_reports: 0,
  top_reactions: [
    { reaction: 'FATIGUE', count: 38026 },
    { reaction: 'NAUSEA', count: 31453 },
    { reaction: 'DYSPNOEA', count: 31828 },
  ],
  source: 'mock',
  fallback: true,
};

const MOCK_TRIALS: TrialsResponse = {
  condition: 'diabetes',
  total_found: 4872,
  trials: [
    { nct_id: 'NCT06172166', title: 'Transdisciplinary Care for Young Adults With Type 1 Diabetes', phase: 'N/A', status: 'RECRUITING', enrollment: 80, start_date: '2024-04-17', locations_count: 2 },
    { nct_id: 'NCT06019624', title: 'Fresh Food Boxes for Diabetes Management', phase: 'N/A', status: 'RECRUITING', enrollment: 400, start_date: '2023-09-13', locations_count: 1 },
  ],
  source: 'mock',
  fallback: true,
};

// ── Service ───────────────────────────────────────────────────────────────────

@Injectable({ providedIn: 'root' })
export class ApiService {
  private http = inject(HttpClient);

  /** Health check — use to verify backend is reachable */
  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${API_BASE}/health`).pipe(
      catchError(() => of({ status: 'unreachable', service: 'healthrisk-ai', version: '?', environment: 'unknown', uptime_seconds: 0 }))
    );
  }

  /** Combined live dashboard data from WHO + FAERS + ClinicalTrials + CMS */
  liveDashboard(): Observable<LiveDashboard> {
    return this.http.get<LiveDashboard>(`${API_BASE}/api/v1/dashboard/live`).pipe(
      catchError(() => of(MOCK_DASHBOARD)),
      shareReplay(1)
    );
  }

  /** WHO GHO life expectancy (global) */
  whoIndicators(): Observable<WhoResponse> {
    return this.http.get<WhoResponse>(`${API_BASE}/api/v1/who/indicators`).pipe(
      catchError(() => of(MOCK_WHO))
    );
  }

  /** WHO GHO diabetes prevalence top countries */
  whoDiseaseBurden(): Observable<WhoResponse> {
    return this.http.get<WhoResponse>(`${API_BASE}/api/v1/who/disease-burden`).pipe(
      catchError(() => of({ ...MOCK_WHO, indicator: 'diabetes_prevalence' }))
    );
  }

  /** FDA FAERS adverse event signals for a drug */
  faersSignals(drug = 'aspirin'): Observable<FaersResponse> {
    const params = new HttpParams().set('drug', drug);
    return this.http.get<FaersResponse>(`${API_BASE}/api/v1/faers/signals`, { params }).pipe(
      catchError(() => of({ ...MOCK_FAERS, drug }))
    );
  }

  /** ClinicalTrials.gov recruiting trials for a condition */
  recruitingTrials(condition = 'diabetes', limit = 10): Observable<TrialsResponse> {
    const params = new HttpParams().set('condition', condition).set('limit', limit);
    return this.http.get<TrialsResponse>(`${API_BASE}/api/v1/trials/recruiting`, { params }).pipe(
      catchError(() => of({ ...MOCK_TRIALS, condition }))
    );
  }

  /** CMS Hospital Compare data */
  cmsHospitals(state = '', limit = 20): Observable<CmsResponse> {
    let params = new HttpParams().set('limit', limit);
    if (state) params = params.set('state', state);
    return this.http.get<CmsResponse>(`${API_BASE}/api/v1/cms/hospitals`, { params }).pipe(
      catchError(() => of({ hospitals: [], source: 'mock', count: 0, fallback: true }))
    );
  }

  /** SEC EDGAR pharma companies */
  edgarPharma(limit = 5): Observable<EdgarResponse> {
    const params = new HttpParams().set('limit', limit);
    return this.http.get<EdgarResponse>(`${API_BASE}/api/v1/edgar/pharma`, { params }).pipe(
      catchError(() => of({ companies: [], source: 'mock', count: 0, fallback: true }))
    );
  }
}
