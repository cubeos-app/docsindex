package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"
)

// Config holds all configuration from environment variables.
type Config struct {
	ListenAddr     string // HTTP listen address
	DocsRepoURL    string // Git repo URL for documentation
	DocsLocalPath  string // Local path to store docs
	OllamaHost     string // Ollama host for embeddings
	OllamaPort     string // Ollama port
	EmbeddingModel string // Model for embeddings (nomic-embed-text)
	ChromaHost     string // ChromaDB host
	ChromaPort     string // ChromaDB port
	CollectionName string // ChromaDB collection name
	SyncInterval   int    // Sync interval in hours (0 = disable periodic sync)
	ChunkSize      int    // Max characters per chunk
	ChunkOverlap   int    // Overlap between chunks
}

func loadConfig() *Config {
	return &Config{
		ListenAddr:     getEnv("LISTEN_ADDR", ":8080"),
		DocsRepoURL:    getEnv("DOCS_REPO_URL", "https://github.com/cubeos-app/docs.git"),
		DocsLocalPath:  getEnv("DOCS_LOCAL_PATH", "/cubeos/docs"),
		OllamaHost:     getEnv("OLLAMA_HOST", ""),
		OllamaPort:     getEnv("OLLAMA_PORT", ""),
		EmbeddingModel: getEnv("EMBEDDING_MODEL", "nomic-embed-text"),
		ChromaHost:     getEnv("CHROMADB_HOST", ""),
		ChromaPort:     getEnv("CHROMADB_PORT", ""),
		CollectionName: getEnv("COLLECTION_NAME", "cubeos_docs"),
		SyncInterval:   getEnvInt("SYNC_INTERVAL_HOURS", 6),
		ChunkSize:      getEnvInt("CHUNK_SIZE", 500),
		ChunkOverlap:   getEnvInt("CHUNK_OVERLAP", 50),
	}
}

func getEnv(key, defaultVal string) string {
	if val := os.Getenv(key); val != "" {
		return val
	}
	return defaultVal
}

func getEnvInt(key string, defaultVal int) int {
	if val := os.Getenv(key); val != "" {
		var i int
		if _, err := fmt.Sscanf(val, "%d", &i); err == nil {
			return i
		}
	}
	return defaultVal
}

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

// DocTreeItem represents a node in the docs file tree.
type DocTreeItem struct {
	Title    string        `json:"title"`
	Path     string        `json:"path"`
	IsDir    bool          `json:"is_dir"`
	Children []DocTreeItem `json:"children,omitempty"`
}

// DocContent is the response for a single document fetch.
type DocContent struct {
	Title   string `json:"title"`
	Path    string `json:"path"`
	Content string `json:"content"`
}

// SearchResult is a single search hit.
type SearchResult struct {
	Title   string  `json:"title"`
	Path    string  `json:"path"`
	Snippet string  `json:"snippet"`
	Score   float32 `json:"score,omitempty"`
}

// IndexStatus reports the current state of the indexer.
type IndexStatus struct {
	LastRun    string `json:"last_run"`
	DocCount   int    `json:"doc_count"`
	ChunkCount int    `json:"chunk_count"`
	Indexing   bool   `json:"indexing"`
	Error      string `json:"error,omitempty"`
}

// ---------------------------------------------------------------------------
// ChromaDB / Ollama wire types
// ---------------------------------------------------------------------------

type ChromaCollection struct {
	ID   string `json:"id"`
	Name string `json:"name"`
}

type ChromaAddRequest struct {
	IDs        []string            `json:"ids"`
	Embeddings [][]float32         `json:"embeddings"`
	Documents  []string            `json:"documents"`
	Metadatas  []map[string]string `json:"metadatas"`
}

type ChromaQueryRequest struct {
	QueryEmbeddings [][]float32 `json:"query_embeddings"`
	NResults        int         `json:"n_results"`
	Include         []string    `json:"include"`
}

type ChromaQueryResponse struct {
	IDs       [][]string            `json:"ids"`
	Documents [][]string            `json:"documents"`
	Metadatas [][]map[string]string `json:"metadatas"`
	Distances [][]float32           `json:"distances"`
}

type OllamaEmbeddingRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type OllamaEmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

// Document represents a chunk of documentation for indexing.
type Document struct {
	ID       string            `json:"id"`
	Content  string            `json:"content"`
	Metadata map[string]string `json:"metadata"`
}

// ---------------------------------------------------------------------------
// Built-in fallback documentation (served when /cubeos/docs is empty)
// ---------------------------------------------------------------------------

var builtinDocs = map[string]DocContent{
	"getting-started": {
		Title: "Getting Started with CubeOS",
		Path:  "getting-started",
		Content: `# Getting Started with CubeOS

Welcome to CubeOS — your self-hosted server operating system for Raspberry Pi.

## Quick Links

- **Dashboard:** [http://cubeos.cube](http://cubeos.cube)
- **API:** [http://api.cubeos.cube](http://api.cubeos.cube)
- **Pi-hole DNS:** [http://pihole.cubeos.cube](http://pihole.cubeos.cube)
- **Logs (Dozzle):** [http://dozzle.cubeos.cube](http://dozzle.cubeos.cube)

## First Steps

1. Connect to the CubeOS WiFi access point
2. Open the dashboard at http://cubeos.cube
3. Complete the Setup Wizard to configure your device
4. Install additional services from the App Store

## Network Modes

- **Offline** — Access Point only, air-gapped operation
- **Online (Ethernet)** — AP + internet via Ethernet cable
- **Online (WiFi)** — AP + internet via USB WiFi dongle

## Documentation

Full documentation is available online at [docs.cubeos.app](https://docs.cubeos.app).

To enable offline documentation with AI-powered search, ensure the Ollama and ChromaDB services are running on the Services page. Documentation will be automatically indexed when these services become available.
`,
	},
}

var builtinTree = []DocTreeItem{
	{
		Title: "Getting Started with CubeOS",
		Path:  "getting-started",
		IsDir: false,
	},
}

// ---------------------------------------------------------------------------
// Server
// ---------------------------------------------------------------------------

type Server struct {
	config *Config
	mu     sync.RWMutex
	status IndexStatus
}

func main() {
	config := loadConfig()

	log.Printf("CubeOS Document Indexer v0.2.0-alpha.01 starting...")
	log.Printf("  Listen:     %s", config.ListenAddr)
	log.Printf("  Docs repo:  %s", config.DocsRepoURL)
	log.Printf("  Local path: %s", config.DocsLocalPath)
	if config.OllamaHost != "" && config.OllamaPort != "" {
		log.Printf("  Ollama:     %s:%s (model: %s)", config.OllamaHost, config.OllamaPort, config.EmbeddingModel)
	} else {
		log.Printf("  Ollama:     not configured")
	}
	if config.ChromaHost != "" && config.ChromaPort != "" {
		log.Printf("  ChromaDB:   %s:%s (collection: %s)", config.ChromaHost, config.ChromaPort, config.CollectionName)
	} else {
		log.Printf("  ChromaDB:   not configured")
	}
	if config.OllamaHost == "" || config.ChromaHost == "" {
		log.Printf("  AI search disabled (Ollama/ChromaDB not configured). Running in filesystem mode.")
	}
	log.Printf("  Sync every: %d hours", config.SyncInterval)

	srv := &Server{config: config}

	// Background: initial indexing + periodic sync
	go srv.backgroundIndexer()

	// HTTP server
	mux := http.NewServeMux()

	// Health
	mux.HandleFunc("/health", srv.handleHealth)

	// Docs API — served under /api/v1/docs to match dashboard expectations
	mux.HandleFunc("/api/v1/docs/status", srv.handleDocsServiceStatus)
	mux.HandleFunc("/api/v1/docs/tree", srv.handleDocsTree)
	mux.HandleFunc("/api/v1/docs/search", srv.handleDocsSearch)
	// Catch-all for /api/v1/docs/{path...}
	mux.HandleFunc("/api/v1/docs/", srv.handleDocsGet)

	// Index status / trigger
	mux.HandleFunc("/api/v1/index/status", srv.handleIndexStatus)
	mux.HandleFunc("/api/v1/index/trigger", srv.handleIndexTrigger)

	log.Printf("HTTP server listening on %s", config.ListenAddr)
	if err := http.ListenAndServe(config.ListenAddr, mux); err != nil {
		log.Fatalf("HTTP server failed: %v", err)
	}
}

// ---------------------------------------------------------------------------
// HTTP handlers
// ---------------------------------------------------------------------------

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "ok"})
}

func (s *Server) handleDocsServiceStatus(w http.ResponseWriter, r *http.Request) {
	// Check if docs directory has content
	docsAvailable := false
	if files, err := findMarkdownFiles(s.config.DocsLocalPath); err == nil && len(files) > 0 {
		docsAvailable = true
	}

	// Check Ollama reachability (only if configured)
	ollamaOK := false
	if s.config.OllamaHost != "" && s.config.OllamaPort != "" {
		ollamaURL := fmt.Sprintf("http://%s:%s", s.config.OllamaHost, s.config.OllamaPort)
		client := &http.Client{Timeout: 3 * time.Second}
		if resp, err := client.Get(ollamaURL); err == nil {
			resp.Body.Close()
			ollamaOK = true
		}
	}

	// Check ChromaDB reachability (only if configured)
	chromaOK := false
	if s.config.ChromaHost != "" && s.config.ChromaPort != "" {
		chromaURL := fmt.Sprintf("http://%s:%s/api/v2", s.config.ChromaHost, s.config.ChromaPort)
		client := &http.Client{Timeout: 3 * time.Second}
		if resp, err := client.Get(chromaURL); err == nil {
			resp.Body.Close()
			chromaOK = true
		}
	}

	s.mu.RLock()
	indexStatus := s.status
	s.mu.RUnlock()

	mode := "builtin"
	if docsAvailable && ollamaOK && chromaOK {
		mode = "rag"
	} else if docsAvailable {
		mode = "filesystem"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"mode":           mode,
		"docs_available": docsAvailable,
		"ollama_ok":      ollamaOK,
		"chromadb_ok":    chromaOK,
		"indexing":       indexStatus.Indexing,
		"last_indexed":   indexStatus.LastRun,
		"doc_count":      indexStatus.DocCount,
		"chunk_count":    indexStatus.ChunkCount,
		"index_error":    indexStatus.Error,
	})
}

func (s *Server) handleDocsTree(w http.ResponseWriter, r *http.Request) {
	tree, err := buildDocsTree(s.config.DocsLocalPath)
	if err != nil {
		log.Printf("Failed to build docs tree: %v, serving built-in docs", err)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(builtinTree)
		return
	}

	// If docs directory is empty, serve built-in fallback
	if len(tree) == 0 {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(builtinTree)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(tree)
}

func (s *Server) handleDocsGet(w http.ResponseWriter, r *http.Request) {
	// Strip prefix to get the doc path
	docPath := strings.TrimPrefix(r.URL.Path, "/api/v1/docs/")
	if docPath == "" {
		docPath = "README"
	}

	// Check built-in docs first (fallback when docs dir is empty)
	if doc, ok := builtinDocs[docPath]; ok {
		// Only serve built-in if no real docs exist at this path
		cleaned := filepath.Clean(docPath)
		fullPath := filepath.Join(s.config.DocsLocalPath, cleaned)
		if !strings.HasSuffix(fullPath, ".md") {
			fullPath += ".md"
		}
		if _, err := os.Stat(fullPath); os.IsNotExist(err) {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(doc)
			return
		}
	}

	// Prevent path traversal
	cleaned := filepath.Clean(docPath)
	if strings.Contains(cleaned, "..") {
		http.Error(w, `{"error":"invalid path"}`, http.StatusBadRequest)
		return
	}

	// Try with .md extension if not present
	fullPath := filepath.Join(s.config.DocsLocalPath, cleaned)
	if !strings.HasSuffix(fullPath, ".md") {
		fullPath += ".md"
	}

	content, err := os.ReadFile(fullPath)
	if err != nil {
		if os.IsNotExist(err) {
			http.Error(w, `{"error":"document not found"}`, http.StatusNotFound)
			return
		}
		http.Error(w, `{"error":"failed to read document"}`, http.StatusInternalServerError)
		return
	}

	title := extractTitle(string(content), filepath.Base(cleaned))

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(DocContent{
		Title:   title,
		Path:    cleaned,
		Content: string(content),
	})
}

func (s *Server) handleDocsSearch(w http.ResponseWriter, r *http.Request) {
	query := r.URL.Query().Get("q")
	if query == "" {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode([]SearchResult{})
		return
	}

	results, err := s.searchChromaDB(query)
	if err != nil {
		log.Printf("ChromaDB search failed, falling back to filesystem: %v", err)
		// Fallback: basic filesystem grep
		results = s.filesystemSearch(query)
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(results)
}

func (s *Server) handleIndexStatus(w http.ResponseWriter, r *http.Request) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(s.status)
}

func (s *Server) handleIndexTrigger(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, `{"error":"POST required"}`, http.StatusMethodNotAllowed)
		return
	}

	s.mu.RLock()
	busy := s.status.Indexing
	s.mu.RUnlock()

	if busy {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusConflict)
		json.NewEncoder(w).Encode(map[string]string{"error": "indexing already in progress"})
		return
	}

	go s.runIndexingCycle()
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]string{"status": "indexing started"})
}

// ---------------------------------------------------------------------------
// Docs tree builder
// ---------------------------------------------------------------------------

func buildDocsTree(root string) ([]DocTreeItem, error) {
	if _, err := os.Stat(root); os.IsNotExist(err) {
		return []DocTreeItem{}, nil
	}

	var items []DocTreeItem
	dirs := make(map[string]*DocTreeItem)

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil // skip errors
		}

		relPath, _ := filepath.Rel(root, path)

		// Skip hidden dirs, .git, etc.
		if info.IsDir() {
			base := filepath.Base(path)
			if strings.HasPrefix(base, ".") || base == "node_modules" {
				return filepath.SkipDir
			}
			return nil
		}

		// Only markdown files
		if !strings.HasSuffix(strings.ToLower(path), ".md") {
			return nil
		}

		// Build path without extension for the API path
		docPath := strings.TrimSuffix(relPath, filepath.Ext(relPath))

		// Read title
		content, err := os.ReadFile(path)
		title := extractTitle(string(content), filepath.Base(docPath))
		if err != nil {
			title = filepath.Base(docPath)
		}

		dir := filepath.Dir(relPath)
		item := DocTreeItem{
			Title: title,
			Path:  docPath,
			IsDir: false,
		}

		if dir == "." {
			// Top-level file
			items = append(items, item)
		} else {
			// Nested in a directory
			dirItem, exists := dirs[dir]
			if !exists {
				dirItem = &DocTreeItem{
					Title:    formatDirName(dir),
					Path:     dir,
					IsDir:    true,
					Children: []DocTreeItem{},
				}
				dirs[dir] = dirItem
			}
			dirItem.Children = append(dirItem.Children, item)
		}

		return nil
	})
	if err != nil {
		return nil, err
	}

	// Sort directories and add to items
	var dirNames []string
	for name := range dirs {
		dirNames = append(dirNames, name)
	}
	sort.Strings(dirNames)

	// Top-level files first (sorted), then directories
	sort.Slice(items, func(i, j int) bool {
		return items[i].Title < items[j].Title
	})

	for _, name := range dirNames {
		dirItem := dirs[name]
		sort.Slice(dirItem.Children, func(i, j int) bool {
			return dirItem.Children[i].Title < dirItem.Children[j].Title
		})
		items = append(items, *dirItem)
	}

	return items, nil
}

func formatDirName(dir string) string {
	base := filepath.Base(dir)
	words := strings.Split(strings.ReplaceAll(base, "-", " "), " ")
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + w[1:]
		}
	}
	return strings.Join(words, " ")
}

// ---------------------------------------------------------------------------
// Filesystem search (fallback when ChromaDB is unavailable)
// ---------------------------------------------------------------------------

func (s *Server) filesystemSearch(query string) []SearchResult {
	query = strings.ToLower(query)
	var results []SearchResult

	filepath.Walk(s.config.DocsLocalPath, func(path string, info os.FileInfo, err error) error {
		if err != nil || info.IsDir() {
			return nil
		}
		if !strings.HasSuffix(strings.ToLower(path), ".md") {
			return nil
		}
		// Skip hidden
		if strings.Contains(path, "/.") {
			return nil
		}

		content, err := os.ReadFile(path)
		if err != nil {
			return nil
		}

		lower := strings.ToLower(string(content))
		if !strings.Contains(lower, query) {
			return nil
		}

		relPath, _ := filepath.Rel(s.config.DocsLocalPath, path)
		docPath := strings.TrimSuffix(relPath, filepath.Ext(relPath))
		title := extractTitle(string(content), filepath.Base(docPath))

		// Extract snippet around first match
		idx := strings.Index(lower, query)
		start := idx - 80
		if start < 0 {
			start = 0
		}
		end := idx + len(query) + 80
		if end > len(content) {
			end = len(content)
		}
		snippet := strings.TrimSpace(string(content[start:end]))

		results = append(results, SearchResult{
			Title:   title,
			Path:    docPath,
			Snippet: snippet,
		})

		return nil
	})

	return results
}

// ---------------------------------------------------------------------------
// ChromaDB search
// ---------------------------------------------------------------------------

func (s *Server) searchChromaDB(query string) ([]SearchResult, error) {
	ollamaURL := fmt.Sprintf("http://%s:%s", s.config.OllamaHost, s.config.OllamaPort)

	// Get embedding for query
	embedding, err := getEmbedding(ollamaURL, s.config.EmbeddingModel, query)
	if err != nil {
		return nil, fmt.Errorf("embedding failed: %w", err)
	}

	// Get collection ID
	collectionID, err := getCollectionID(s.config)
	if err != nil {
		return nil, fmt.Errorf("collection lookup failed: %w", err)
	}

	// Query ChromaDB
	baseURL := fmt.Sprintf("http://%s:%s/api/v2", s.config.ChromaHost, s.config.ChromaPort)
	queryURL := fmt.Sprintf("%s/tenants/default_tenant/databases/default_database/collections/%s/query",
		baseURL, collectionID)

	queryReq := ChromaQueryRequest{
		QueryEmbeddings: [][]float32{embedding},
		NResults:        10,
		Include:         []string{"documents", "metadatas", "distances"},
	}
	body, _ := json.Marshal(queryReq)

	resp, err := http.Post(queryURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("query request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("query failed (status %d): %s", resp.StatusCode, string(respBody))
	}

	var chromaResp ChromaQueryResponse
	if err := json.NewDecoder(resp.Body).Decode(&chromaResp); err != nil {
		return nil, fmt.Errorf("decode response failed: %w", err)
	}

	// Convert to search results, dedup by source path
	seen := make(map[string]bool)
	var results []SearchResult

	if len(chromaResp.IDs) > 0 {
		for i, id := range chromaResp.IDs[0] {
			_ = id
			meta := chromaResp.Metadatas[0][i]
			source := meta["source"]
			docPath := strings.TrimSuffix(source, filepath.Ext(source))

			if seen[docPath] {
				continue
			}
			seen[docPath] = true

			snippet := ""
			if i < len(chromaResp.Documents[0]) {
				snippet = chromaResp.Documents[0][i]
				if len(snippet) > 200 {
					snippet = snippet[:200] + "..."
				}
			}

			var score float32
			if i < len(chromaResp.Distances[0]) {
				score = chromaResp.Distances[0][i]
			}

			results = append(results, SearchResult{
				Title:   meta["title"],
				Path:    docPath,
				Snippet: snippet,
				Score:   score,
			})
		}
	}

	return results, nil
}

func getCollectionID(config *Config) (string, error) {
	baseURL := fmt.Sprintf("http://%s:%s/api/v2", config.ChromaHost, config.ChromaPort)
	url := fmt.Sprintf("%s/tenants/default_tenant/databases/default_database/collections/%s",
		baseURL, config.CollectionName)

	resp, err := http.Get(url)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return "", fmt.Errorf("collection not found (status %d)", resp.StatusCode)
	}

	var coll ChromaCollection
	if err := json.NewDecoder(resp.Body).Decode(&coll); err != nil {
		return "", err
	}
	return coll.ID, nil
}

// ---------------------------------------------------------------------------
// Background indexer
// ---------------------------------------------------------------------------

func (s *Server) backgroundIndexer() {
	// Wait a moment for HTTP server to come up
	time.Sleep(2 * time.Second)

	// Initial run
	s.runIndexingCycle()

	if s.config.SyncInterval == 0 {
		log.Println("Sync interval is 0, background indexer disabled")
		return
	}

	ticker := time.NewTicker(time.Duration(s.config.SyncInterval) * time.Hour)
	defer ticker.Stop()

	for range ticker.C {
		log.Println("Starting scheduled sync...")
		s.runIndexingCycle()
	}
}

func (s *Server) runIndexingCycle() {
	s.mu.Lock()
	s.status.Indexing = true
	s.status.Error = ""
	s.mu.Unlock()

	docCount, chunkCount, err := runIndexing(s.config)

	s.mu.Lock()
	s.status.Indexing = false
	s.status.LastRun = time.Now().UTC().Format(time.RFC3339)
	s.status.DocCount = docCount
	s.status.ChunkCount = chunkCount
	if err != nil {
		s.status.Error = err.Error()
		log.Printf("ERROR: Indexing failed: %v", err)
	} else {
		log.Printf("Indexing complete: %d docs, %d chunks", docCount, chunkCount)
	}
	s.mu.Unlock()
}

// ---------------------------------------------------------------------------
// Indexing pipeline
// ---------------------------------------------------------------------------

func runIndexing(config *Config) (docCount, chunkCount int, err error) {
	// Step 1: Sync docs from Git
	log.Println("Step 1: Syncing documentation from Git...")
	if err := syncDocs(config); err != nil {
		// Non-fatal: docs may already exist locally
		log.Printf("  Git sync warning: %v (continuing with local files)", err)
	}

	// Step 2: Find markdown files
	log.Println("Step 2: Finding markdown files...")
	mdFiles, err := findMarkdownFiles(config.DocsLocalPath)
	if err != nil {
		return 0, 0, fmt.Errorf("failed to find markdown files: %w", err)
	}
	log.Printf("  Found %d markdown files", len(mdFiles))
	docCount = len(mdFiles)

	if len(mdFiles) == 0 {
		log.Println("No markdown files found, nothing to index")
		return docCount, 0, nil
	}

	// Step 3: Parse and chunk
	log.Println("Step 3: Parsing and chunking documents...")
	documents, err := parseAndChunkDocuments(mdFiles, config)
	if err != nil {
		return docCount, 0, fmt.Errorf("failed to parse documents: %w", err)
	}
	chunkCount = len(documents)
	log.Printf("  Created %d chunks", chunkCount)

	// Step 4: Ensure collection exists (create if missing, don't delete)
	log.Println("Step 4: Ensuring ChromaDB collection...")
	collectionID, err := ensureCollection(config)
	if err != nil {
		log.Printf("  ChromaDB unavailable: %v (docs served from filesystem only)", err)
		return docCount, chunkCount, nil
	}
	log.Printf("  Collection ID: %s", collectionID)

	// Step 5: Generate embeddings and store
	log.Println("Step 5: Generating embeddings and storing in ChromaDB...")
	if err := indexDocuments(documents, collectionID, config); err != nil {
		log.Printf("  Indexing to ChromaDB failed: %v (docs served from filesystem only)", err)
		return docCount, chunkCount, nil
	}

	return docCount, chunkCount, nil
}

func syncDocs(config *Config) error {
	gitDir := filepath.Join(config.DocsLocalPath, ".git")
	if _, err := os.Stat(gitDir); err == nil {
		// Existing repo — pull
		log.Println("  Pulling updates...")
		cmd := exec.Command("git", "-C", config.DocsLocalPath, "pull", "--ff-only")
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			log.Println("  Pull failed, resetting to origin...")
			exec.Command("git", "-C", config.DocsLocalPath, "fetch", "origin").Run()
			exec.Command("git", "-C", config.DocsLocalPath, "reset", "--hard", "origin/main").Run()
		}
	} else if config.DocsRepoURL != "" {
		// Clone to temp dir first — NEVER destroy existing docs (B45)
		// /cubeos/docs/ may contain packer-seeded content that must survive
		// a failed clone (e.g. offline mode, repo unreachable).
		tmpDir := config.DocsLocalPath + ".clone-tmp"
		os.RemoveAll(tmpDir) // clean stale temp only
		log.Printf("  Cloning %s to temp dir...", config.DocsRepoURL)
		if err := os.MkdirAll(filepath.Dir(config.DocsLocalPath), 0755); err != nil {
			return fmt.Errorf("failed to create parent dir: %w", err)
		}
		cmd := exec.Command("git", "clone", "--depth=1", config.DocsRepoURL, tmpDir)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Run(); err != nil {
			os.RemoveAll(tmpDir)
			return fmt.Errorf("git clone failed: %w", err)
		}
		// Clone succeeded — atomically swap directories
		backupDir := config.DocsLocalPath + ".bak"
		os.RemoveAll(backupDir)
		if err := os.Rename(config.DocsLocalPath, backupDir); err != nil {
			// Target may not exist (first run), that's OK
			log.Printf("  Note: could not backup existing docs: %v", err)
		}
		if err := os.Rename(tmpDir, config.DocsLocalPath); err != nil {
			// Restore backup if swap fails
			os.Rename(backupDir, config.DocsLocalPath)
			os.RemoveAll(tmpDir)
			return fmt.Errorf("failed to swap docs directory: %w", err)
		}
		os.RemoveAll(backupDir)
		log.Println("  Git clone succeeded, docs swapped in")
	}
	return nil
}

func findMarkdownFiles(root string) ([]string, error) {
	var files []string
	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return nil
		}
		// Skip hidden dirs
		if info.IsDir() && strings.HasPrefix(filepath.Base(path), ".") {
			return filepath.SkipDir
		}
		if !info.IsDir() && strings.HasSuffix(strings.ToLower(path), ".md") {
			files = append(files, path)
		}
		return nil
	})
	return files, err
}

func parseAndChunkDocuments(files []string, config *Config) ([]Document, error) {
	var documents []Document

	for _, file := range files {
		content, err := os.ReadFile(file)
		if err != nil {
			log.Printf("  Warning: Could not read %s: %v", file, err)
			continue
		}

		relPath, _ := filepath.Rel(config.DocsLocalPath, file)
		title := extractTitle(string(content), filepath.Base(file))
		chunks := chunkText(string(content), config.ChunkSize, config.ChunkOverlap)

		for i, chunk := range chunks {
			hash := sha256.Sum256([]byte(file + fmt.Sprintf("%d", i) + chunk))
			id := hex.EncodeToString(hash[:8])

			documents = append(documents, Document{
				ID:      id,
				Content: chunk,
				Metadata: map[string]string{
					"source": relPath,
					"title":  title,
					"chunk":  fmt.Sprintf("%d", i),
					"total":  fmt.Sprintf("%d", len(chunks)),
				},
			})
		}
	}

	return documents, nil
}

func extractTitle(content, filename string) string {
	lines := strings.Split(content, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# ") {
			return strings.TrimPrefix(line, "# ")
		}
	}
	return strings.TrimSuffix(filename, filepath.Ext(filename))
}

func chunkText(text string, chunkSize, overlap int) []string {
	text = strings.ReplaceAll(text, "\r\n", "\n")
	paragraphs := strings.Split(text, "\n\n")

	var chunks []string
	var currentChunk strings.Builder

	for _, para := range paragraphs {
		para = strings.TrimSpace(para)
		if para == "" {
			continue
		}

		if currentChunk.Len()+len(para) > chunkSize && currentChunk.Len() > 0 {
			chunks = append(chunks, strings.TrimSpace(currentChunk.String()))

			prevContent := currentChunk.String()
			currentChunk.Reset()
			if len(prevContent) > overlap {
				currentChunk.WriteString(prevContent[len(prevContent)-overlap:])
				currentChunk.WriteString("\n\n")
			}
		}

		currentChunk.WriteString(para)
		currentChunk.WriteString("\n\n")
	}

	if currentChunk.Len() > 0 {
		chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
	}

	if len(chunks) == 0 && len(text) > 0 {
		chunks = append(chunks, strings.TrimSpace(text))
	}

	return chunks
}

// ensureCollection gets or creates the ChromaDB collection (non-destructive).
func ensureCollection(config *Config) (string, error) {
	baseURL := fmt.Sprintf("http://%s:%s/api/v2", config.ChromaHost, config.ChromaPort)
	tenant := "default_tenant"
	database := "default_database"

	// Try to get existing collection
	url := fmt.Sprintf("%s/tenants/%s/databases/%s/collections/%s", baseURL, tenant, database, config.CollectionName)

	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return "", fmt.Errorf("failed to reach ChromaDB: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == 200 {
		var collection ChromaCollection
		if err := json.NewDecoder(resp.Body).Decode(&collection); err == nil {
			return collection.ID, nil
		}
	}

	// Create new collection
	log.Println("  Creating collection...")
	createURL := fmt.Sprintf("%s/tenants/%s/databases/%s/collections", baseURL, tenant, database)
	createBody := map[string]interface{}{
		"name": config.CollectionName,
	}
	bodyBytes, _ := json.Marshal(createBody)

	resp, err = client.Post(createURL, "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		return "", fmt.Errorf("failed to create collection: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 && resp.StatusCode != 201 {
		body, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("create collection failed: %s", string(body))
	}

	var collection ChromaCollection
	if err := json.NewDecoder(resp.Body).Decode(&collection); err != nil {
		return "", fmt.Errorf("failed to decode collection: %w", err)
	}

	return collection.ID, nil
}

func indexDocuments(documents []Document, collectionID string, config *Config) error {
	if len(documents) == 0 {
		return nil
	}

	baseURL := fmt.Sprintf("http://%s:%s/api/v2", config.ChromaHost, config.ChromaPort)
	ollamaURL := fmt.Sprintf("http://%s:%s", config.OllamaHost, config.OllamaPort)

	batchSize := 10
	for i := 0; i < len(documents); i += batchSize {
		end := i + batchSize
		if end > len(documents) {
			end = len(documents)
		}
		batch := documents[i:end]

		var ids []string
		var embeddings [][]float32
		var contents []string
		var metadatas []map[string]string

		for j, doc := range batch {
			log.Printf("  Embedding %d/%d: %s (chunk %s)",
				i+j+1, len(documents), doc.Metadata["source"], doc.Metadata["chunk"])

			embedding, err := getEmbedding(ollamaURL, config.EmbeddingModel, doc.Content)
			if err != nil {
				log.Printf("  Warning: Embedding failed for %s: %v", doc.ID, err)
				continue
			}

			ids = append(ids, doc.ID)
			embeddings = append(embeddings, embedding)
			contents = append(contents, doc.Content)
			metadatas = append(metadatas, doc.Metadata)
		}

		if len(ids) == 0 {
			continue
		}

		addURL := fmt.Sprintf("%s/tenants/default_tenant/databases/default_database/collections/%s/add",
			baseURL, collectionID)
		addReq := ChromaAddRequest{
			IDs:        ids,
			Embeddings: embeddings,
			Documents:  contents,
			Metadatas:  metadatas,
		}
		bodyBytes, _ := json.Marshal(addReq)

		resp, err := http.Post(addURL, "application/json", bytes.NewReader(bodyBytes))
		if err != nil {
			return fmt.Errorf("failed to add to ChromaDB: %w", err)
		}
		resp.Body.Close()

		if resp.StatusCode != 200 && resp.StatusCode != 201 {
			return fmt.Errorf("ChromaDB add failed (status %d)", resp.StatusCode)
		}

		log.Printf("  Stored batch of %d", len(ids))
	}

	return nil
}

func getEmbedding(ollamaURL, model, text string) ([]float32, error) {
	reqBody := OllamaEmbeddingRequest{
		Model:  model,
		Prompt: text,
	}
	bodyBytes, _ := json.Marshal(reqBody)

	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Post(ollamaURL+"/api/embeddings", "application/json", bytes.NewReader(bodyBytes))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama error: %s", string(body))
	}

	var embResp OllamaEmbeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&embResp); err != nil {
		return nil, fmt.Errorf("decode failed: %w", err)
	}

	return embResp.Embedding, nil
}
