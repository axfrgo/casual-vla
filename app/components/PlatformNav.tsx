'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { ArrowRight, Shield } from 'lucide-react';
import { DINNER_TABLE_MISSION, missionHref, workspaceHref } from '../lib/mission';

type PlatformNavProps = {
  missionSeed?: number;
  missionStep?: number;
};

export function PlatformNav({ missionSeed, missionStep }: PlatformNavProps) {
  const pathname = usePathname();
  const onMission = pathname === '/';
  const onWorkspace = pathname.startsWith('/aegis');
  const missionLink = missionHref(missionSeed, missionStep);
  const workspaceLink = workspaceHref(missionSeed, missionStep);
  const evidenceLink = `/?seed=${missionSeed ?? 1001}#evidence`;

  return (
    <header className="platform-nav">
      <Link className="platform-brand" href={missionLink} aria-label="Fortifiers Aegis mission home">
        <Shield size={22} />
        <span>FORTIFIERS</span>
        <b>AEGIS / 01</b>
      </Link>
      <div className="platform-nav-center">
        <span className="platform-nav-eyebrow">COMPETITION CONTROL PLANE</span>
        <nav aria-label="Platform navigation">
          <Link className={onMission ? 'active' : ''} href={missionLink} aria-current={onMission ? 'page' : undefined}>
            <small>01</small> Mission
          </Link>
          <Link className={onWorkspace ? 'active' : ''} href={workspaceLink} aria-current={onWorkspace ? 'page' : undefined}>
            <small>02</small> Teach &amp; remember
          </Link>
          <Link className="nav-evidence" href={evidenceLink}>
            <small>03</small> Evidence
          </Link>
        </nav>
      </div>
      <div className="platform-nav-status">
        <span className="platform-live-dot" />
        <span>{onWorkspace ? 'WORKSPACE SYNCED' : 'BROWSER READY'}</span>
        <strong>{DINNER_TABLE_MISSION.shortId} · {missionSeed ?? 1001}</strong>
        {onMission && <Link href={workspaceLink} aria-label="Open the teaching workspace"><ArrowRight size={15} /></Link>}
      </div>
    </header>
  );
}
