import { Inter } from 'next/font/google'
import './globals.css'
import ErrorBoundary from '../components/ErrorBoundary'

const inter = Inter({ subsets: ['latin'] })

export const metadata = {
    title: 'ResearchVisionPro — Research-Grade Image Processing',
    description:
        'Professional image processing tool for academic researchers. ' +
        'Spatial filtering, frequency domain analysis, morphological operations, ' +
        'segmentation, color processing, deep learning inference, and remote sensing.',
}

export default function RootLayout({ children }) {
    return (
        <html lang="en" className="dark">
            <body
                className={`${inter.className} bg-gray-900 text-white min-h-screen`}
            >
                <ErrorBoundary>
                    {children}
                </ErrorBoundary>
            </body>
        </html>
    )
}
