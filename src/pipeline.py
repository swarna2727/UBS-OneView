"""Simplified Investment Strategy Pipeline Orchestrator."""

import os
import json
import uuid
import time
from typing import Dict, Any, Optional
from datetime import datetime

from src.utils.logger import setup_logger
from src.agents.agent_1 import InvestmentIdeaGenerator
from src.agents.agent_2 import SignalModellingSpecialist
from src.agents.agent_3 import StrategyDefinitionEngineer
from src.agents.agent_4 import RiskComplianceValidator
from src.agents.agent_5 import InvestmentMemoWriter
from src.utils.pdf_generator import PDFGenerator

class InvestmentPipeline:
    """Simplified orchestrator for the sequential AI agent pipeline."""
    
    def __init__(self, api_key: str, data_dir: str = "data/pipeline_data"):
        self.api_key = api_key
        self.data_dir = data_dir
        self.logger = setup_logger("InvestmentPipeline")
        self.pdf_generator = PDFGenerator()
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize agents
        self.agents = [
            InvestmentIdeaGenerator(api_key),
            SignalModellingSpecialist(api_key),
            StrategyDefinitionEngineer(api_key),
            RiskComplianceValidator(api_key),
            InvestmentMemoWriter(api_key)
        ]

    # --- Shared schema helpers for collaborative rounds ---
    def _ensure_shared_schema(self, data: Dict[str, Any]) -> None:
        if 'schema' not in data:
            data['schema'] = {}

    def _get_or_create_path(self, root: Dict[str, Any], path: list) -> Dict[str, Any]:
        node = root
        for key in path:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        return node

    def _merge_with_metadata(self, data: Dict[str, Any], path: list, value: Any, proposed_by: str, comment: Optional[str] = None, confidence: float = 0.7) -> None:
        """Merge a field into data['schema'] with PRD-style metadata and history."""
        self._ensure_shared_schema(data)
        target = self._get_or_create_path(data['schema'], path[:-1])
        field_key = path[-1]

        existing = target.get(field_key, {}) if isinstance(target.get(field_key), dict) else {}

        # Preserve history
        history = existing.get('history', []) if isinstance(existing, dict) else []
        if 'value' in existing:
            history.append({
                'value': existing.get('value'),
                'proposed_by': existing.get('proposed_by'),
                'timestamp': datetime.now().isoformat()
            })

        # Build comments
        comments = existing.get('comments', [])
        if comment:
            comments = comments + [{ 'agent': proposed_by, 'text': comment }]

        # Write merged node
        target[field_key] = {
            'value': value,
            'proposed_by': proposed_by,
            'confidence': round(confidence, 2),
            'comments': comments,
            'history': history
        }
    
    def _initialize_schema(self, input_data: Dict[str, Any], transaction_id: str) -> Dict[str, Any]:
        """Initialize the master JSON schema with input data."""
        return {
            "metadata": {
                "transaction_id": transaction_id,
                "timestamp": datetime.now().isoformat(),
                "total_processing_time_ms": 0
            },
            "input": input_data,
            "pdf_path": ""
        }
    
    def _save_data(self, data: Dict[str, Any], transaction_id: str) -> str:
        """Save data to JSON file."""
        file_path = os.path.join(self.data_dir, f"{transaction_id}_complete.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        return file_path
    
    def run(self, input_data: Dict[str, Any], transaction_id: Optional[str] = None, rounds: int = 2) -> Dict[str, Any]:
        """
        Run the complete investment pipeline.
        
        Args:
            input_data: Initial input data (market_conditions, investment_themes, constraints)
            transaction_id: Optional transaction ID
            
        Returns:
            Complete data with all agent outputs and PDF path
        """
        if transaction_id is None:
            transaction_id = str(uuid.uuid4())
        
        start_time = time.time()
        self.logger.info(f"Starting pipeline with transaction ID: {transaction_id}")
        
        try:
            # Initialize master schema
            data = self._initialize_schema(input_data, transaction_id)

            # Determine core agents (1-4) and committee agent (5)
            core_agents = self.agents[:-1]
            committee_agent = self.agents[-1]

            # Collaborative rounds
            rounds = max(1, int(rounds))
            previous_round_schema_snapshot = None
            for round_num in range(1, rounds + 1):
                round_start = time.time()
                self.logger.info(f"=== Starting collaboration round {round_num}/{rounds} ===")
                for i, agent in enumerate(core_agents, 1):
                    self.logger.info(f"Running Agent {i} (Round {round_num}): {agent.name}")
                    
                    # Special handling: run Agent 2 per idea and aggregate results
                    if agent.__class__.__name__ == 'SignalModellingSpecialist':
                        try:
                            a1 = data.get('agent_1_output', {})
                            ideas = a1.get('investment_ideas', [])
                            by_idea = {}
                            per_call_times = []
                            for idea in ideas:
                                idea_id = idea.get('idea_id')
                                if not idea_id:
                                    continue
                                # Set current idea as selected for this sub-run
                                data.setdefault('agent_1_output', {})['selected_idea_id'] = idea_id
                                sub_result = agent.run(data)
                                # Extract agent_2_output
                                a2 = sub_result.get('agent_2_output', {})
                                by_idea[idea_id] = {
                                    'signals': a2.get('signals', []),
                                    'failure_modes': a2.get('failure_modes', []),
                                    'backtesting_requirements': a2.get('backtesting_requirements', {})
                                }
                                # Capture time if present
                                if isinstance(a2.get('processing_time_ms'), (int, float)):
                                    per_call_times.append(a2.get('processing_time_ms'))
                                # Keep data up to date for next steps
                                data = sub_result
                            # Aggregate into a unified structure
                            data['agent_2_output'] = {
                                'by_idea': by_idea,
                                'processing_time_ms': sum(per_call_times) if per_call_times else 0,
                                'additional_fields': {}
                            }
                        except Exception as per_idea_err:
                            self.logger.error(f"Agent 2 per-idea aggregation failed: {str(per_idea_err)}")
                            # Fallback to single run
                            data = agent.run(data)
                    else:
                        data = agent.run(data)


                    # Merge agent outputs into shared schema with metadata (heuristic confidence)
                    try:
                        if agent.__class__.__name__ == 'InvestmentIdeaGenerator':
                            a1 = data.get('agent_1_output', {})
                            sel_id = a1.get('selected_idea_id')
                            selected = None
                            for idea in a1.get('investment_ideas', []):
                                if idea.get('idea_id') == sel_id:
                                    selected = idea
                                    break
                            if selected:
                                self._merge_with_metadata(
                                    data,
                                    ['strategy', 'name'],
                                    selected.get('strategy_name'),
                                    agent.name,
                                    comment=f"Round {round_num}: generated and selected idea {sel_id}",
                                    confidence=0.75
                                )
                                self._merge_with_metadata(
                                    data,
                                    ['strategy', 'description'],
                                    selected.get('description'),
                                    agent.name,
                                    comment=None,
                                    confidence=0.72
                                )
                        elif agent.__class__.__name__ == 'SignalModellingSpecialist':
                            a2 = data.get('agent_2_output', {})
                            by_idea = a2.get('by_idea', {})
                            # Store the entire per-idea modeling bundle
                            self._merge_with_metadata(
                                data,
                                ['modeling', 'by_idea'],
                                by_idea,
                                agent.name,
                                comment=f"Round {round_num}: signals for {len(by_idea)} ideas",
                                confidence=0.7
                            )
                        elif agent.__class__.__name__ == 'StrategyDefinitionEngineer':
                            a3 = data.get('agent_3_output', {})
                            self._merge_with_metadata(
                                data,
                                ['strategy_rules', 'definition'],
                                a3.get('strategy_rules', {}),
                                agent.name,
                                comment=f"Round {round_num}: provided rules and pseudocode",
                                confidence=0.73
                            )
                        elif agent.__class__.__name__ == 'RiskComplianceValidator':
                            a4 = data.get('agent_4_output', {})
                            self._merge_with_metadata(
                                data,
                                ['risk', 'compliance'],
                                a4.get('risk_compliance', {}),
                                agent.name,
                                comment=f"Round {round_num}: identified risk & compliance controls",
                                confidence=0.7
                            )
                    except Exception as merge_err:
                        self.logger.warning(f"Metadata merge skipped due to error: {str(merge_err)}")

                    # Save per-agent snapshot with new naming: agent and round prefix, tx suffix
                    agent_prefix = f"agent{i}_{agent.__class__.__name__}_round{round_num}"
                    intermediate_path = os.path.join(self.data_dir, f"{agent_prefix}_{transaction_id}.json")
                    with open(intermediate_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    # Save per-round snapshot (single file per round, after each agent overwrite with same naming)
                    round_path = os.path.join(self.data_dir, f"round{round_num}_{transaction_id}.json")
                    with open(round_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    self.logger.info(f"Saved snapshots to {intermediate_path} and {round_path}")

                # Round timing and convergence check
                round_duration_ms = round((time.time() - round_start) * 1000, 2)
                self.logger.info(f"=== Completed round {round_num} in {round_duration_ms}ms ===")

                # Early stop on convergence (no change in shared schema compared to previous round)
                try:
                    import copy, json as _json
                    current_schema = data.get('schema', {})
                    current_schema_str = _json.dumps(current_schema, sort_keys=True)
                    if previous_round_schema_snapshot is not None and current_schema_str == previous_round_schema_snapshot:
                        self.logger.info(f"No schema changes detected after round {round_num}. Early stopping.")
                        break
                    previous_round_schema_snapshot = current_schema_str
                except Exception as conv_err:
                    self.logger.warning(f"Convergence check failed: {str(conv_err)}")

            # Final synthesis by Investment Committee (Agent 5)
            self.logger.info("Running Investment Committee synthesis (Agent 5)")
            data = committee_agent.run(data)

            # Generate PDF memo
            self.logger.info("Generating PDF investment memo")
            pdf_path = self.pdf_generator.generate_investment_memo_pdf(
                data,
                os.path.join(self.data_dir, f"{transaction_id}_final_memo.pdf")
            )
            data['pdf_path'] = pdf_path
            
            # Calculate total processing time
            total_time = round((time.time() - start_time) * 1000, 2)
            data['metadata']['total_processing_time_ms'] = total_time
            
            # Save complete data
            json_path = self._save_data(data, transaction_id)

            # PRD-compatible aliases
            try:
                # schema_final.json alias
                schema_final_path = os.path.join(self.data_dir, 'schema_final.json')
                with open(schema_final_path, 'w') as f:
                    json.dump(data, f, indent=2)
                # transaction-scoped schema alias
                schema_final_tx_path = os.path.join(self.data_dir, f"schema_final_{transaction_id}.json")
                with open(schema_final_tx_path, 'w') as f:
                    json.dump(data, f, indent=2)

                # final_report.txt alias
                report_text = data.get('agent_5_output', {}).get('report_text', '')
                report_txt_path = os.path.join(self.data_dir, 'final_report.txt')
                with open(report_txt_path, 'w', encoding='utf-8') as f:
                    f.write(report_text or '')
                # transaction-scoped report alias
                report_txt_tx_path = os.path.join(self.data_dir, f"final_report_{transaction_id}.txt")
                with open(report_txt_tx_path, 'w', encoding='utf-8') as f:
                    f.write(report_text or '')
                self.logger.info(f"Exported PRD aliases: {schema_final_path}, {schema_final_tx_path}, {report_txt_path}, {report_txt_tx_path}")
            except Exception as export_err:
                self.logger.warning(f"Failed to export PRD aliases: {str(export_err)}")
            
            self.logger.info(f"Pipeline completed successfully in {total_time}ms")
            
            return {
                'success': True,
                'transaction_id': transaction_id,
                'total_processing_time_ms': total_time,
                'pdf_path': pdf_path,
                'json_path': json_path,
                'data': data
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'transaction_id': transaction_id
            }
    
    def get_pipeline_status(self, transaction_id: str) -> Dict[str, Any]:
        """Get status of a pipeline run."""
        json_path = os.path.join(self.data_dir, f"{transaction_id}_complete.json")
        pdf_path = os.path.join(self.data_dir, f"{transaction_id}_final_memo.pdf")
        
        status = {
            'transaction_id': transaction_id,
            'json_exists': os.path.exists(json_path),
            'pdf_exists': os.path.exists(pdf_path),
            'json_path': json_path if os.path.exists(json_path) else None,
            'pdf_path': pdf_path if os.path.exists(pdf_path) else None
        }
        
        if status['json_exists']:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                status['agents_completed'] = []
                for i in range(1, 6):
                    if f'agent_{i}_output' in data:
                        status['agents_completed'].append(i)
                status['total_processing_time_ms'] = data.get('metadata', {}).get('total_processing_time_ms', 0)
            except Exception as e:
                status['error'] = f"Error reading JSON: {str(e)}"
        
        return status