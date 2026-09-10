import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'Fortifiers Apprenticeship | Aegis', description:'Teach machines the way you teach people. An auditable operational teaching workspace for contamination-control research.' };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
