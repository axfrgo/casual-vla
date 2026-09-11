import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'Fortifiers Aegis | Competition Mission', description:'A deterministic browser view of the Fortifiers dual-arm dinner-table challenge, with native MuJoCo evidence kept explicit and separate.' };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
