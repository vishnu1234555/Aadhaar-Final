import { useState } from 'react'
import axios from 'axios'
import './index.css'

function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState('')

  const handleExtract = async () => {
    if (!text.trim()) return

    setLoading(true)
    setError('')
    setResults(null)

    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:5000';
      const response = await axios.post(`${apiUrl}/api/extract`, {
        text: text,
        threshold: 0.4
      })

      if (response.data.success) {
        setResults({
          entities: response.data.entities,
          time: response.data.inference_time_ms
        })
      } else {
        setError(response.data.error || 'Failed to extract entities.')
      }
    } catch (err) {
      console.error(err)
      setError(
        err.response?.data?.error || 
        'Is the backend running? Ensure you started the Flask server on port 5000.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="header">
        <h1>Identity Intelligence</h1>
        <p>AI-powered entity extraction for Aadhaar & Banking data</p>
      </header>

      {/* Input Section */}
      <section className="glass-card input-section">
        <h2>
          <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
          </svg>
          Source Text
        </h2>
        
        <textarea 
          placeholder="Paste unstructured document text, ID details, or banking logs here..."
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={loading}
        />
        
        <button 
          className="action-btn" 
          onClick={handleExtract}
          disabled={loading || text.trim().length === 0}
        >
          {loading ? (
            <>
              <div className="loader"></div>
              Extracting...
            </>
          ) : (
            'Extract Entities'
          )}
        </button>

        {error && <div className="error-msg">{error}</div>}
      </section>

      {/* Results Section */}
      <section className="glass-card results-section">
        <h2>
          <span>
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" style={{ verticalAlign: 'middle', marginRight: '8px' }}>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"></path>
            </svg>
            Extracted Intelligence
          </span>
          {results && <span className="time-badge">{results.time}ms</span>}
        </h2>

        {!results && !loading && (
          <div className="empty-state">
            <svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" strokeWidth="1" style={{ opacity: 0.5 }}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"></path>
            </svg>
            <p>Intelligence pending.<br/>Submit text to analyze identities.</p>
          </div>
        )}

        {loading && (
          <div className="empty-state" style={{ border: 'none' }}>
            <div className="loader" style={{ width: '40px', height: '40px', borderWidth: '4px', borderColor: 'rgba(109,40,217,0.2)', borderTopColor: 'var(--primary)' }}></div>
            <p style={{ marginTop: '1rem', color: 'var(--primary)' }}>Analyzing content...</p>
          </div>
        )}

        {results && results.entities && results.entities.length === 0 && (
          <div className="empty-state" style={{ borderColor: 'rgba(239, 68, 68, 0.3)' }}>
            <p>No valid entities detected in the text.</p>
          </div>
        )}

        {results && results.entities && results.entities.length > 0 && (
          <div className="entities-list">
            {results.entities.map((entity, index) => (
              <div key={index} className="entity-item">
                <div className="entity-header">
                  <span className="entity-label">{entity.label}</span>
                  <span className="entity-score">{entity.confidence}%</span>
                </div>
                <div className="entity-value">{entity.text}</div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}

export default App
