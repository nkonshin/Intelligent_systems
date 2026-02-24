"""
Экспертная система диагностики простудных заболеваний
Практическое занятие №3 - Прототип ядра ЭС

Дата: Февраль 2026
"""

# Патч для совместимости с Python 3.10+
import sys
import collections
import collections.abc
for name in ['Mapping', 'MutableMapping', 'Iterable', 'MutableSet', 'Callable']:
    if not hasattr(collections, name):
        setattr(collections, name, getattr(collections.abc, name))

from experta import *
import json
from datetime import datetime


# ============================================================================
# ЭТАП 1: КЛАССЫ ФАКТОВ (Fact Classes)
# ============================================================================

class Patient(Fact):
    """Факт о пациенте"""
    pass


class Symptom(Fact):
    """Факт о симптоме"""
    pass


class Diagnosis(Fact):
    """Факт о диагнозе"""
    pass


class RiskLevel(Fact):
    """Факт об уровне риска"""
    pass


class Recommendation(Fact):
    """Факт о рекомендации"""
    pass


class FiredRule(Fact):
    """Факт о сработавшем правиле (для объяснений)"""
    pass


# ============================================================================
# ЭТАП 2: ДВИЖОК ЭКСПЕРТНОЙ СИСТЕМЫ
# ============================================================================

class MedicalDiagnosisEngine(KnowledgeEngine):
    """
    Ядро экспертной системы диагностики простудных заболеваний
    
    Реализует 7 ключевых правил:
    - R001: Диагностика гриппа типа A
    - R002: Диагностика ОРВИ
    - R003: Диагностика COVID-19
    - R006: Диагностика легкой простуды
    - R005: Выявление пневмонии (осложнение)
    - R012: Критическая ситуация
    - R008: Высокий риск у пожилых
    """
    
    def __init__(self):
        super().__init__()
        self.fired_rules = []  # Список сработавших правил для объяснений
        self.explanations = []  # Объяснения
    
    def add_explanation(self, rule_id, rule_name, reasoning):
        """Добавить объяснение сработавшего правила"""
        self.fired_rules.append({
            'id': rule_id,
            'name': rule_name,
            'reasoning': reasoning,
            'timestamp': datetime.now().strftime('%H:%M:%S')
        })
        self.explanations.append(f"[{rule_id}] {rule_name}: {reasoning}")
    
    # ========================================================================
    # ПРАВИЛО R001: Диагностика гриппа типа A
    # ========================================================================
    
    @Rule(
        Symptom(name='температура', value=P(lambda x: isinstance(x, (int, float)) and x > 38.5)),
        Symptom(name='начало', value='резкое'),
        Symptom(name='ломота_в_мышцах', value='высокая'),
        Symptom(name='слабость', value='выраженная'),
        NOT(Diagnosis())
    )
    def rule_001_flu_type_a(self):
        """R001: Определение гриппа типа A (CF: 0.85)"""
        self.declare(Diagnosis(
            name='Грипп типа A',
            confidence=0.85,
            severity='средняя'
        ))
        self.declare(Recommendation(
            category='лечение',
            text='Постельный режим, противовирусные препараты (осельтамивир), обильное питье',
            priority='высокий'
        ))
        self.add_explanation(
            'R001',
            'Диагностика гриппа типа A',
            'Высокая температура (>38.5°C) + резкое начало + сильная ломота в мышцах + выраженная слабость указывают на грипп типа A'
        )
    
    # ========================================================================
    # ПРАВИЛО R002: Диагностика ОРВИ
    # ========================================================================
    
    @Rule(
        Symptom(name='температура', value=P(lambda x: isinstance(x, (int, float)) and 37.5 <= x <= 38)),
        Symptom(name='насморк', value=True),
        Symptom(name='боль_в_горле', value='умеренная'),
        Symptom(name='начало', value='постепенное'),
        NOT(Diagnosis())
    )
    def rule_002_common_cold(self):
        """R002: Определение ОРВИ (CF: 0.90)"""
        self.declare(Diagnosis(
            name='ОРВИ',
            confidence=0.90,
            severity='легкая'
        ))
        self.declare(Recommendation(
            category='лечение',
            text='Симптоматическое лечение, покой, теплое питье, промывание носа',
            priority='средний'
        ))
        self.add_explanation(
            'R002',
            'Диагностика ОРВИ',
            'Умеренная температура (37.5-38°C) + насморк + боль в горле + постепенное начало характерны для ОРВИ'
        )
    
    # ========================================================================
    # ПРАВИЛО R003: Диагностика COVID-19
    # ========================================================================
    
    @Rule(
        OR(
            Symptom(name='потеря_обоняния', value=True),
            Symptom(name='потеря_вкуса', value=True)
        ),
        Symptom(name='температура', value=P(lambda x: isinstance(x, (int, float)) and x > 37.5)),
        OR(
            Symptom(name='кашель', value='сухой'),
            Symptom(name='одышка', value=MATCH.any_value)
        ),
        NOT(Diagnosis())
    )
    def rule_003_covid19(self):
        """R003: Выявление COVID-19 (CF: 0.75)"""
        self.declare(Diagnosis(
            name='Подозрение на COVID-19',
            confidence=0.75,
            severity='средняя'
        ))
        self.declare(Recommendation(
            category='диагностика',
            text='СРОЧНО сдать ПЦР-тест, самоизоляция, мониторинг сатурации',
            priority='критический'
        ))
        self.add_explanation(
            'R003',
            'Выявление COVID-19',
            'Потеря обоняния/вкуса + повышенная температура + сухой кашель/одышка - признаки COVID-19'
        )
    
    # ========================================================================
    # ПРАВИЛО R006: Диагностика легкой простуды
    # ========================================================================
    
    @Rule(
        Symptom(name='температура', value=P(lambda x: isinstance(x, (int, float)) and x < 37.5)),
        Symptom(name='насморк', value=True),
        Symptom(name='недомогание', value='легкое'),
        NOT(Symptom(name='головная_боль', value='сильная')),
        NOT(Diagnosis())
    )
    def rule_006_mild_cold(self):
        """R006: Неосложненная простуда (CF: 0.95)"""
        self.declare(Diagnosis(
            name='Легкая простуда',
            confidence=0.95,
            severity='легкая'
        ))
        self.declare(Recommendation(
            category='лечение',
            text='Домашнее лечение, витамин C, теплое питье, отдых',
            priority='низкий'
        ))
        self.add_explanation(
            'R006',
            'Диагностика легкой простуды',
            'Нормальная температура (<37.5°C) + легкий насморк + отсутствие тяжелых симптомов'
        )
    
    # ========================================================================
    # ПРАВИЛО R005: Выявление пневмонии
    # ========================================================================
    
    @Rule(
        Symptom(name='длительность_заболевания', value=P(lambda x: isinstance(x, (int, float)) and x > 7)),
        Symptom(name='температура', value=P(lambda x: isinstance(x, (int, float)) and x > 38)),
        Symptom(name='кашель_с_мокротой', value=True),
        Symptom(name='боль_в_груди', value=True),
        Symptom(name='одышка', value=MATCH.any_value)
        # ИСПРАВЛЕНИЕ №1: Убрано NOT(Diagnosis()) - пневмония может сосуществовать с критическим состоянием
    )
    def rule_005_pneumonia(self):
        """R005: Выявление риска пневмонии (CF: 0.70)"""
        self.declare(Diagnosis(
            name='Риск пневмонии',
            confidence=0.70,
            severity='тяжелая'
        ))
        self.declare(Recommendation(
            category='госпитализация',
            text='СРОЧНО обратиться к врачу, рентген грудной клетки, возможна госпитализация',
            priority='критический'
        ))
        self.add_explanation(
            'R005',
            'Выявление пневмонии',
            'Длительное заболевание (>7 дней) + высокая температура + кашель с мокротой + боль в груди + одышка'
        )
    
    # ========================================================================
    # ПРАВИЛО R012: Критическая ситуация
    # ========================================================================
    
    @Rule(
        OR(
            Symptom(name='одышка', value='тяжелая'),
            Symptom(name='сатурация_O2', value=P(lambda x: isinstance(x, (int, float)) and x < 92)),
            AND(
                Symptom(name='температура', value=P(lambda x: isinstance(x, (int, float)) and x > 40)),
                Symptom(name='жаропонижающие_помогают', value=False)
            ),
            Symptom(name='спутанность_сознания', value=True)
        )
    )
    def rule_012_critical_emergency(self):
        """R012: Критические симптомы (CF: 1.0)"""
        self.declare(Diagnosis(
            name='КРИТИЧЕСКОЕ СОСТОЯНИЕ',
            confidence=1.0,
            severity='критическая'
        ))
        self.declare(Recommendation(
            category='экстренная_помощь',
            text='НЕМЕДЛЕННЫЙ вызов скорой помощи (103), транспортировка в стационар',
            priority='экстренный'
        ))
        self.add_explanation(
            'R012',
            'Критическая ситуация',
            'Обнаружены критические симптомы, требующие немедленной госпитализации'
        )
    
    # ========================================================================
    # ПРАВИЛО R008: Высокий риск у пожилых
    # ========================================================================
    
    @Rule(
        Patient(age=P(lambda x: isinstance(x, (int, float)) and x > 65)),
        OR(
            Diagnosis(name='Грипп типа A'),
            Diagnosis(name='ОРВИ')
        ),
        Patient(chronic_diseases=True)
    )
    def rule_008_elderly_risk(self):
        """R008: Высокий риск осложнений у пожилых (CF: 0.88)"""
        self.declare(RiskLevel(
            level='высокий',
            confidence=0.88,
            reason='пожилой_возраст_с_хроническими_заболеваниями'
        ))
        self.declare(Recommendation(
            category='мониторинг',
            text='Обязательная консультация врача, ежедневный мониторинг, противовирусная терапия',
            priority='высокий'
        ))
        self.add_explanation(
            'R008',
            'Группа риска: пожилые',
            'Возраст >65 лет + грипп/ОРВИ + хронические заболевания = высокий риск осложнений'
        )
    
    # ========================================================================
    # Вспомогательные методы
    # ========================================================================
    
    def get_fired_rules(self):
        """Получить список сработавших правил"""
        return self.fired_rules
    
    def get_explanations(self):
        """Получить объяснения"""
        return self.explanations
    
    def print_results(self):
        """Вывести результаты диагностики"""
        print("\n" + "="*70)
        print("                    РЕЗУЛЬТАТЫ ДИАГНОСТИКИ")
        print("="*70)
        
        # Диагнозы
        diagnoses = []
        for fact in self.facts.values():
            if isinstance(fact, Diagnosis):
                diagnoses.append(fact)
        
        if diagnoses:
            print("\n📋 УСТАНОВЛЕННЫЕ ДИАГНОЗЫ:")
            for diag in sorted(diagnoses, key=lambda x: x.get('confidence', 0), reverse=True):
                conf_percent = int(diag.get('confidence', 0) * 100)
                severity = diag.get('severity', 'неизвестная')
                print(f"\n  • {diag['name']}")
                print(f"    Уверенность: {conf_percent}% | Тяжесть: {severity}")
        else:
            print("\n❌ Диагноз не установлен. Недостаточно данных.")
        
        # Уровень риска
        risks = [f for f in self.facts.values() if isinstance(f, RiskLevel)]
        if risks:
            print("\n⚠️  УРОВЕНЬ РИСКА:")
            for risk in risks:
                conf_percent = int(risk.get('confidence', 0) * 100)
                print(f"  • {risk['level'].upper()} ({conf_percent}%)")
                print(f"    Причина: {risk.get('reason', 'не указана')}")
        
        # Рекомендации
        recommendations = [f for f in self.facts.values() if isinstance(f, Recommendation)]
        if recommendations:
            print("\n💊 РЕКОМЕНДАЦИИ:")
            # Сортировка по приоритету
            priority_order = {'экстренный': 0, 'критический': 1, 'высокий': 2, 'средний': 3, 'низкий': 4}
            sorted_recs = sorted(
                recommendations, 
                key=lambda x: priority_order.get(x.get('priority', 'средний'), 3)
            )
            
            for rec in sorted_recs:
                category = rec.get('category', 'общее').upper()
                priority = rec.get('priority', 'средний')
                text = rec.get('text', '')
                
                # Эмодзи в зависимости от приоритета
                emoji = {
                    'экстренный': '🚨',
                    'критический': '⚠️',
                    'высокий': '❗',
                    'средний': '💊',
                    'низкий': '✓'
                }.get(priority, '•')
                
                print(f"\n  {emoji} [{category}] (приоритет: {priority})")
                print(f"    {text}")
        
        # Сработавшие правила (объяснения)
        print("\n" + "="*70)
        print("                    ОБЪЯСНЕНИЕ РАССУЖДЕНИЙ")
        print("="*70)
        
        if self.fired_rules:
            print(f"\nСработало правил: {len(self.fired_rules)}\n")
            for i, rule in enumerate(self.fired_rules, 1):
                print(f"{i}. [{rule['id']}] {rule['name']}")
                print(f"   Время: {rule['timestamp']}")
                print(f"   Обоснование: {rule['reasoning']}")
                print()
        else:
            print("\nПравила не сработали. Проверьте входные данные.")
        
        print("="*70 + "\n")


# ============================================================================
# ЭТАП 2: ИНТЕРАКТИВНЫЙ ИНТЕРФЕЙС
# ============================================================================

def interactive_input():
    """Интерактивный ввод симптомов через консоль"""
    print("\n" + "="*70)
    print("     ЭКСПЕРТНАЯ СИСТЕМА ДИАГНОСТИКИ ПРОСТУДНЫХ ЗАБОЛЕВАНИЙ")
    print("="*70)
    print("\nРежим интерактивного ввода. Ответьте на вопросы.\n")
    
    engine = MedicalDiagnosisEngine()
    engine.reset()
    
    # Данные пациента
    print("--- ДАННЫЕ ПАЦИЕНТА ---\n")
    age = int(input("Возраст пациента: "))
    chronic = input("Есть ли хронические заболевания? (да/нет): ").lower() == 'да'
    
    engine.declare(Patient(age=age, chronic_diseases=chronic))
    
    # Основные симптомы
    print("\n--- ОСНОВНЫЕ СИМПТОМЫ ---\n")
    
    temp = float(input("Температура тела (°C): "))
    engine.declare(Symptom(name='температура', value=temp))
    
    onset = input("Характер начала (резкое/постепенное): ").lower()
    if onset in ['резкое', 'постепенное']:
        engine.declare(Symptom(name='начало', value=onset))
    
    # Респираторные симптомы
    print("\n--- РЕСПИРАТОРНЫЕ СИМПТОМЫ ---\n")
    
    has_runny_nose = input("Есть насморк? (да/нет): ").lower() == 'да'
    if has_runny_nose:
        engine.declare(Symptom(name='насморк', value=True))
    
    cough = input("Кашель (нет/сухой/с мокротой): ").lower()
    if cough == 'сухой':
        engine.declare(Symptom(name='кашель', value='сухой'))
    elif cough == 'с мокротой':
        engine.declare(Symptom(name='кашель_с_мокротой', value=True))
    
    throat_pain = input("Боль в горле (нет/легкая/умеренная/сильная): ").lower()
    if throat_pain in ['легкая', 'умеренная', 'сильная']:
        engine.declare(Symptom(name='боль_в_горле', value=throat_pain))
    
    # Общие симптомы
    print("\n--- ОБЩИЕ СИМПТОМЫ ---\n")
    
    weakness = input("Слабость (нет/легкая/умеренная/выраженная): ").lower()
    if weakness in ['выраженная', 'умеренная', 'легкая']:
        engine.declare(Symptom(name='слабость', value=weakness))
    
    muscle_pain = input("Ломота в мышцах (нет/легкая/умеренная/высокая): ").lower()
    if muscle_pain in ['высокая', 'умеренная', 'легкая']:
        engine.declare(Symptom(name='ломота_в_мышцах', value=muscle_pain))
    
    malaise = input("Общее недомогание (нет/легкое/умеренное/сильное): ").lower()
    if malaise == 'легкое':
        engine.declare(Symptom(name='недомогание', value='легкое'))
    
    # Специфические симптомы
    print("\n--- СПЕЦИФИЧЕСКИЕ СИМПТОМЫ ---\n")
    
    loss_smell = input("Потеря обоняния? (да/нет): ").lower() == 'да'
    if loss_smell:
        engine.declare(Symptom(name='потеря_обоняния', value=True))
    
    loss_taste = input("Потеря вкуса? (да/нет): ").lower() == 'да'
    if loss_taste:
        engine.declare(Symptom(name='потеря_вкуса', value=True))
    
    # Дополнительные данные
    print("\n--- ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ---\n")
    
    duration = input("Длительность заболевания (дней, или 0 если только началось): ")
    if duration.isdigit() and int(duration) > 0:
        engine.declare(Symptom(name='длительность_заболевания', value=int(duration)))
    
    chest_pain = input("Боль в груди? (да/нет): ").lower() == 'да'
    if chest_pain:
        engine.declare(Symptom(name='боль_в_груди', value=True))
    
    # Критические симптомы
    print("\n--- КРИТИЧЕСКИЕ СИМПТОМЫ (при наличии) ---\n")
    
    dyspnea = input("Одышка (нет/легкая/умеренная/тяжелая): ").lower()
    if dyspnea in ['тяжелая', 'умеренная', 'легкая']:
        engine.declare(Symptom(name='одышка', value=dyspnea))
    
    if temp > 38:
        fever_meds = input("Помогают ли жаропонижающие? (да/нет): ").lower() == 'да'
        engine.declare(Symptom(name='жаропонижающие_помогают', value=fever_meds))
    
    return engine


def load_scenario(scenario_file):
    """Загрузить тестовый сценарий из JSON"""
    with open(scenario_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    engine = MedicalDiagnosisEngine()
    engine.reset()
    
    # Загрузить факты
    for fact_data in data.get('facts', []):
        # Создаем копию словаря, чтобы не изменять оригинал
        fact_dict = dict(fact_data)
        fact_type = fact_dict.pop('type')
        if fact_type == 'Patient':
            engine.declare(Patient(**fact_dict))
        elif fact_type == 'Symptom':
            engine.declare(Symptom(**fact_dict))
    
    return engine, data.get('description', 'Без описания')


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция - меню выбора режима"""
    print("\n" + "="*70)
    print("     ЭКСПЕРТНАЯ СИСТЕМА ДИАГНОСТИКИ ПРОСТУДНЫХ ЗАБОЛЕВАНИЙ")
    print("                      Прототип ядра ЭС v1.0")
    print("="*70)
    
    print("\nВыберите режим работы:")
    print("1. Интерактивный ввод симптомов")
    print("2. Тестовый сценарий #1 (Грипп у пожилого пациента)")
    print("3. Тестовый сценарий #2 (COVID-19)")
    print("4. Тестовый сценарий #3 (Критическая ситуация)")
    print("0. Выход")
    
    choice = input("\nВаш выбор: ")
    
    if choice == '1':
        engine = interactive_input()
        print("\n🔄 Запуск механизма вывода...")
        engine.run()
        engine.print_results()
    
    elif choice == '2':
        engine, desc = load_scenario('scenario_1_elderly_flu.json')
        print(f"\n📋 Загружен сценарий: {desc}")
        print("🔄 Запуск механизма вывода...")
        engine.run()
        engine.print_results()
    
    elif choice == '3':
        engine, desc = load_scenario('scenario_2_covid.json')
        print(f"\n📋 Загружен сценарий: {desc}")
        print("🔄 Запуск механизма вывода...")
        engine.run()
        engine.print_results()
    
    elif choice == '4':
        engine, desc = load_scenario('scenario_3_critical.json')
        print(f"\n📋 Загружен сценарий: {desc}")
        print("🔄 Запуск механизма вывода...")
        engine.run()
        engine.print_results()
    
    elif choice == '0':
        print("\nДо свидания!")
        return
    
    else:
        print("\n❌ Неверный выбор. Попробуйте снова.")
        main()


if __name__ == '__main__':
    main()
