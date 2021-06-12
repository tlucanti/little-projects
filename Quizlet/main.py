from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, ElementNotInteractableException, \
    StaleElementReferenceException, ElementClickInterceptedException
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.support.ui import Select
import time

# 31 - red
# 32 - green
# 33 - yellow


time_start = time.time()
with open('log', 'r+') as logfile:
    CHROME_DRIVER = logfile.readline()
    LOGIN = logfile.readline()
    PASSWORD = logfile.readline()
    MODULE_LINKS = logfile.read().split('\n')
    MODULE_LINKS.pop()
    if len(MODULE_LINKS) != 0:
        print('change modules? (y/n)')
        change = input().strip()
    if len(MODULE_LINKS) == 0 or change == 'y':
        print('insert module links (empty string to exit)')
        print('links should look like this: "https://quizlet.com/ru/123456789/"')
        ml = input('>>> ').strip()
        while ml != '':
            logfile.write(ml)
            logfile.write('\n')
            MODULE_LINKS.append(ml)
            ml = input('>>> ').strip()


def start_driver(chrome_driver_path="/media/kostya/D/PyCharm/test/chromedriver_87"):
    """
    webdriver init function, creates webdriver object to use it next in program
    use webdriver file of your version of chrome
    :param chrome_driver_path: (str) absolute path of folder with webdriver file
    :return: (webdriver) webdriver object
    """
    chrome_options = webdriver.ChromeOptions()
    # chrome_options.add_argument('--headless')
    chrome_options.add_argument('start-maximized')
    chrome_options.add_argument('disable-infobars')
    chrome_options.add_argument('--disable-extensions')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    caps = DesiredCapabilities().CHROME
    # print(chrome_driver_path)
    caps["pageLoadStrategy"] = "none"
    return webdriver.Chrome(chrome_driver_path, options=chrome_options, desired_capabilities=caps)


def find_elem(by, driver, name):
    """
    finding first located element on page with given name/id/class name
    or waiting until element loaded on page
    catching all different exceptions of unsuccessful finding
    cleaning excess spaces
    :param by: (str) element find strategy
    :param driver: (webdriver) webdriver object
    :param name: (str) name/id/class name of html element
    :return: (html element) found element
    """
    by = by.strip()
    name = name.strip()
    time.sleep(ACTION_DELAY)
    if by == 'id':
        find_function = driver.find_element_by_id
    elif by == 'name':
        find_function = driver.find_element_by_name
    elif by == 'class name':
        find_function = driver.find_element_by_class_name
    else:
        raise AttributeError('wrong element find strategy')
    while True:
        try:
            element = find_function(name)
            break
        except NoSuchElementException:
            time.sleep(FIND_DELAY)
        except StaleElementReferenceException:
            time.sleep(FIND_DELAY)
        except ElementNotInteractableException:
            time.sleep(FIND_DELAY)
    while not element.is_displayed():
        time.sleep(FIND_DELAY)
    return element


def find_elems(by, driver, name):
    """
    finding all located elements on page with given names/ids/class names
    or waiting until all elements loaded on page
    not catching any exceptions of unsuccessful finding
    cleaning excess spaces
    :param by: (str) elements find strategy
    :param driver: (webdriver) webdriver object
    :param name: (str) name/id/class name of html elements
    :return: (list) of (html element) found elements
    """
    by = by.strip()
    name = name.strip()
    time.sleep(ACTION_DELAY)
    if by == 'ids':
        find_function = driver.find_elements_by_id
    elif by == 'names':
        find_function = driver.find_elements_by_name
    elif by == 'class names':
        find_function = driver.find_elements_by_class_name
    else:
        raise AttributeError('wrong element find strategy')
    while True:
        elements = find_function(name)
        el_num = len(elements)
        if el_num == 0:
            time.sleep(FIND_DELAY)
            time.sleep(FIND_DELAY)
        else:
            time.sleep(FIND_DELAY)
            elements = find_function(name)
            if len(elements) != el_num:
                time.sleep(FIND_DELAY)
            else:
                return elements


def login_google(login, password, _driver_):
    """
    login with google account on quizlet
    :param login: (str)
    :param password: (str)
    :param _driver_: (webdriver) webdriver object
    :return: (int) status code (0 of ok, else 1)
    """
    # _driver_.maximize_window()
    _driver_.get('https://quizlet.com/login')
    # time.sleep(5)
    find_elem('class name', _driver_, "UISocialButton-labelWrapper").click()
    # print('clicked google')
    find_elem('id', _driver_, "identifierId").send_keys(login + "\n")
    # print('login typed')
    # find_elem('id', _driver_, "identifierNext").click()
    # print('clicked to pass')
    while True:
        try:
            find_elem('name', _driver_, "password").send_keys(password + '\n')
            break
        except StaleElementReferenceException:
            time.sleep(ACTION_DELAY)
    # print('pass typed')
    # print('clicked commit')
    time.sleep(LOGIN_DELAY)
    if _driver_.current_url.startswith('https://accounts.google.com/'):
        print('\033[1;33m[ ! ] CONFIRM LOGIN TO GOOGLE ACCOUNT WITH YOUR PHONE\033[0m')
        while not _driver_.current_url.startswith('https://quizlet.com'):
            time.sleep(LOGIN_DELAY)
    if _driver_.current_url.startswith('https://quizlet.com'):
        print('\033[1;32m[ V ] LOGIN SUCCESS\033[0m')
        return 0
    else:
        print('\033[1;31m[ X ] LOGIN FAILURE\033[0m')
        return 1


def get_words(driver, _mod_hash):
    """
    get all words and translations/explanations in module
    :param driver: (webdriver) webdriver object
    :param _mod_hash: (str) hash number of module (from URL)
    :return: (dict), (dict) forward (RU -> EN) and reverse (EN -> RU) dictionaries
    """
    window_size = driver.get_window_size()
    # driver.maximize_window()
    driver.get('https://quizlet.com/' + _mod_hash)
    time.sleep(3)
    driver.get('https://quizlet.com/' + _mod_hash)
    if len(driver.find_elements_by_class_name('StudyValueUpsellModalHeader')) != 0:
        driver.get('https://quizlet.com/' + _mod_hash)
        time.sleep(SURF_DELAY)
    if len(driver.find_elements_by_class_name('UIModal-closeIcon')) != 0:
        driver.get('https://quizlet.com/' + _mod_hash)
        time.sleep(SURF_DELAY)
    time.sleep(3)
    driver.execute_script("document.body.style.zoom='75%'")
    time.sleep(0.5)
    dropdown_menu = Select(find_elem('class name', driver, 'UIDropdown-select'))
    time.sleep(ACTION_DELAY)
    dropdown_menu.select_by_value('alphabetical')
    time.sleep(1)
    try:
        find_elem('class name', driver, 'SuggestedSetsContentCardGroup-header').click()
    except StaleElementReferenceException:
        time.sleep(SURF_DELAY)
    words_boxes = find_elems('class names', driver, 'SetPageTerm-contentWrapper')
    # driver.set_window_size(height=window_size['height'], width=window_size['width'])
    words_dict = {}  # EN -> RU
    words_back_dict = {}  # RU -> EN
    words_num = len(words_boxes)
    for el in range(words_num):
        # print(el, '$', words_boxes[el].text, '$')
        print('[', '#' * (el * __n // (words_num - 1)), ' ' * (__n - el * __n // (words_num - 1)), ']', '\r',
              sep='', end='')
        text_split = words_boxes[el].text.split('\n')
        text_split = [text_split[0].strip(), text_split[1].strip()]
        words_back_dict[text_split[0]] = text_split[1]
        words_dict[text_split[1]] = text_split[0]
    print()
    return words_dict, words_back_dict


def flash_cards(driver, _words_num, _mod_hash):
    """
    FLASH CARDS IMPLEMENTATION
    to stop running - go to previous page or press Ctrl+C (first is preferable)
    :param driver: (webdriver) webdriver object
    :param _words_num: (int) number of words in module
    :param _mod_hash: (str) hash number of module (from URL)
    :return: (int) status code (0 if ok, else 1)
    """
    click_delay = 0.2
    driver.get('https://quizlet.com/' + _mod_hash + '/flashcards')
    time.sleep(SURF_DELAY)
    while True:
        try:
            continue_button = driver.find_elements_by_class_name('CardsNavigationButton')[1]
            break
        except NoSuchElementException:
            time.sleep(FIND_DELAY)
    try:
        for i in range(_words_num):
            print('[', '#' * (i * __n // (_words_num - 1)), ' ' * (__n - i * __n // (_words_num - 1)), ']', '\r',
                  sep='', end='')
            if not driver.current_url.endswith('flashcards'):
                print()
                print('\033[1;33m[ ! ] FLASHCARDS INTERRUPTED\033[0m')
                return 1
            continue_button.click()
            time.sleep(click_delay)
        if not driver.current_url.endswith('flashcards'):
            print()
            print('\033[1;33m[ ! ] FLASHCARDS INTERRUPTED\033[0m')
            return 1
        print()
        return 0
    except (StaleElementReferenceException, NoSuchElementException, KeyboardInterrupt):
        print()
        print('\033[1;33m[ ! ] FLASHCARDS INTERRUPTED\033[0m')
        return 1


def learn(driver, _words, _back_words, _mod_hash):
    """
    LEARN IMPLEMENTATION
    to stop running - go to previous page or press Ctrl+C (first is preferable)
    :param driver: webdriver object
    :param _words: RU -> EN dict
    :param _back_words: EN -> RU dict
    :param _mod_hash: module hash number (from URL)
    :return: (int) status code (0 if ok, else 1)
    """
    click_delay = 1.5
    # driver.get('https://quizlet.com/' + _mod_hash + '/study-path')
    # while True:
    #     try:
    #         ponyatno = driver.find_elements_by_class_name('UIButton')
    #         if ponyatno[-1].text == 'Понятно':
    #             ponyatno[-1].click()
    #             break
    #     except NoSuchElementException:
    #         time.sleep(FIND_DELAY)
    # find_elem('class name', driver, 'StudyPathIntakeView-skip').click()
    driver.get('https://quizlet.com/' + _mod_hash + '/learn')
    time.sleep(1)
    got_it = driver.find_elements_by_class_name('OnboardingView-gotItButton')
    if len(got_it) != 0:
        got_it[0].find_element_by_class_name('UIButton').click()
    time.sleep(SURF_DELAY)
    __i = 0
    __l = len(_back_words) * 2
    try:
        while True:
            time.sleep(ACTION_DELAY)
            print('[', '#' * (__i * __n // (__l - 1)), ' ' * (__n - __i * __n // (__l - 1)), ']', '\r', sep='', end='')
            __i += 1
            if not driver.current_url.endswith('learn'):
                print()
                print('\033[1;33m[ ! ] LEARN INTERRUPTED\033[0m')
                return 1
            time.sleep(click_delay)
            text_area = driver.find_elements_by_class_name('AutoExpandTextarea-textarea')
            if len(text_area) == 1:
                # >> KEYBOARD MODE
                # >>
                # print('--- KEYBOARD MODE')
                text_area = text_area[0]
                request = find_elem('class name', driver, 'FormattedText').text.strip()
                response = _back_words.get(request, None)
                if response is None:
                    response = _words[request]
                text_area.send_keys(response + '\n')
                # is_error = driver.find_elements_by_class_name('FixedActionLayout')
                # if len(is_error) == 1:
                #     _ = input('\033[1;33m[ ! ] CANNOT FIND RIGHT ANSWER, DO IT BY YOURSELF (PRESS ENTER TO '
                #               'CONTINUE)\033[0m')
                #     continue
            elif len(text_area) == 0:
                buttons = driver.find_elements_by_class_name('MultipleChoiceQuestionPrompt-termOptionInner')
                if len(buttons) == 4:
                    # >> CHOOSE MODE
                    # >>
                    # print('--- choose mode')
                    text = find_elem('class name', driver, 'FormattedText').text.strip()
                    find_text1 = _back_words.get(text, '')
                    find_text2 = _words.get(text, '')
                    for b in buttons:
                        # print(b.text)
                        if b.text == find_text1 or b.text == find_text2:
                            b.click()
                            # print('clicked')
                            break
                    # is_error = driver.find_elements_by_class_name('FixedActionLayout')
                    # if len(is_error) == 1:
                    #     _ = input('\033[1;33m[ ! ] CANNOT FIND RIGHT ANSWER, DO IT BY YOURSELF (PRESS ENTER TO '
                    #               'CONTINUE)\033[0m')
                    #     continue
                else:
                    finish_span = driver.find_elements_by_class_name('EndView-finish')
                    if len(finish_span) != 0:
                        # >> FINISHED
                        # >>
                        # print('--- finished')
                        print()
                        return 0
                    checkpoint_continue_button = driver.find_elements_by_class_name('FixedContinueButton')
                    if len(checkpoint_continue_button) == 1:
                        # >> CHECKPOINT
                        # >>
                        # print('--- checkpoint')
                        checkpoint_continue_button[0].click()
            else:
                print()
                print("--- error")
                continue
    except (StaleElementReferenceException, NoSuchElementException, KeyboardInterrupt):
        print()
        print('\033[1;33m[ ! ] LEARN INTERRUPTED\033[0m')
        return 1


def spell(driver, _words, _mod_hash):
    """
    SPELL IMPLEMENTATION
    to stop running - go to previous page or press Ctrl+C (first is preferable)
    :param driver: webdriver object
    :param _words: RU -> EN dict
    :param _mod_hash: module hash number (from URL)
    :return: (int) status code (0 if ok, else 1)
    """
    click_delay = 1
    right_response = None
    driver.get('https://quizlet.com/' + _mod_hash + '/spell')
    time.sleep(3)
    # __i = 0
    # __l = len(_words) * 2
    try:
        while True:
            # print('[', '#' * (__i * __n // __l + 1), ' ' * (__n - __i * __n // __l - 1), ']', '\r', sep='', end='')
            # __i += 1
            if not driver.current_url.endswith('spell'):
                print()
                print('\033[1;33m[ ! ] SPELL INTERRUPTED\033[0m')
                return 1
            time.sleep(click_delay)
            text_area = driver.find_elements_by_class_name('AutoExpandTextarea-textarea')
            if len(text_area) == 1:
                # >> KEYBOARD MODE
                # >>
                # print('--- KEYBOARD MODE')
                text_area = text_area[0]
                request = find_elem('class name', driver, 'SpellQuestionView-inputPrompt--plain').text.strip()
                if right_response is not None:
                    response = right_response
                    right_response = None
                else:
                    response = _words[request]
                text_area.send_keys(response + '\n')
            elif len(text_area) == 0:
                checkpoint_span = driver.find_elements_by_class_name('SpellCheckpointView-header')
                if len(checkpoint_span) == 1:
                    # >> CHECKPOINT
                    # >>
                    # print('--- CHECKPOINT')
                    find_elem('class name', driver, 'SpellCheckpointView-countdown').click()
                else:
                    spell_correction = driver.find_elements_by_class_name('SpellCorrectionView-diffText')
                    if len(spell_correction) != 0:
                        # >> SPELL CORRECTION
                        # >>
                        # print('--- SPELL CORRECTION')
                        right_response = spell_correction[0].text
                        # print(right_response)
                        driver.find_element_by_class_name('SpellCorrectionView-advanceButton').click()
                    else:
                        finish_span = driver.find_elements_by_class_name('SpellAnalysisView')
                        if len(finish_span) != 0:
                            # >> FINISHED
                            # >>
                            # print('--- finished')
                            print()
                            return 0
            else:
                print()
                print('--- error')
                continue
            progress = driver.find_element_by_class_name('SpellControls-progressValue').text.strip()
            progress_formatted = 0
            for i in progress:
                if i.isdigit():
                    progress_formatted = progress_formatted * 10 + int(i)
            print('[', '#' * (progress_formatted * __n // 100 + 1),
                  ' ' * (__n - (progress_formatted * __n // 100)), ']', '\r', sep='', end='')
    except (StaleElementReferenceException, NoSuchElementException, KeyboardInterrupt):
        print()
        print('\033[1;33m[ ! ] SPELL INTERRUPTED\033[0m')
        return 1


def test(driver, _words, _back_words, _mod_hash):
    """
    TEST IMPLEMENTATION
    to stop running - go to previous page or press Ctrl+C (first is preferable)
    :param driver: webdriver object
    :param _words: RU -> EN dict
    :param _back_words: EN -> RU dict
    :param _mod_hash: module hash number (from URL)
    :return: (int) status code (0 if ok, else 1)
    """
    driver.get('https://quizlet.com/{}/test'.format(_mod_hash))
    segment_delay = 0.2
    time.sleep(3)
    driver.execute_script("document.body.style.zoom='75%'")
    try:
        # write
        write_text_boxes = find_elems('class names', driver, 'AutoExpandTextarea-textarea')
        write_web_elements = find_elems('class names', driver, 'TestModeWrittenQuestion')
        write_number = len(write_web_elements)
        __l = write_number * 4
        __i = 0
        for i in range(write_number):
            text = _back_words.get(write_web_elements[i].text.split('\n')[0].strip(), None)
            if text is None:
                text = _words[write_web_elements[i].text.split('\n')[0].strip()]
            write_text_boxes[i].send_keys(text)
            time.sleep(ACTION_DELAY)
            print('[', '#' * (__i * __n // __l + 1), ' ' * (__n - __i * __n // __l - 1), ']', '\r', sep='', end='')
            __i += 1
        time.sleep(segment_delay)
        # match
        match_text_boxes = find_elems('class names', driver, 'UIInput-input')[write_number:]
        match_web_elements = find_elems('class names', driver, 'TestModeMatchingQuestion-optionsSideListItem')
        match_requests = find_elems('class names', driver, 'TestModeMatchingQuestion-prompt')
        match_responses = [_back_words.get(el.text.strip(), None) for el in match_web_elements]
        if match_responses[0] is None:
            match_responses = [_words[el.text.strip()] for el in match_web_elements]
        match_number = len(match_text_boxes)
        alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        for question in range(match_number):
            for ans in range(match_number):
                if match_requests[question].text == match_responses[ans]:
                    match_text_boxes[question].send_keys(alphabet[ans])
                    break
            print('[', '#' * (__i * __n // __l + 1), ' ' * (__n - __i * __n // __l - 1), ']', '\r', sep='', end='')
            __i += 1
        time.sleep(segment_delay)
        # choice
        choice_web_elements = find_elems('class names', driver, 'TestModeMultipleChoiceQuestion-prompt')
        choice_buttons = find_elems('class names', driver, 'TestModeMultipleChoiceQuestion-choiceText')
        choice_number = len(choice_web_elements)
        answers_per_question = len(choice_buttons) // choice_number
        for question in range(choice_number):
            response = _back_words.get(choice_web_elements[question].text.strip(), None)
            if response is None:
                response = _words[choice_web_elements[question].text.strip()]
            for button in range(answers_per_question):
                if choice_buttons[question * answers_per_question + button].text.strip() == response:
                    to_click = choice_buttons[question * answers_per_question + button]
                    driver.execute_script("arguments[0].click();", to_click)
                    time.sleep(ACTION_DELAY)
                    break
            print('[', '#' * (__i * __n // __l + 1), ' ' * (__n - __i * __n // __l - 1), ']', '\r', sep='', end='')
            __i += 1
        time.sleep(segment_delay)
        # true-false
        true_false_web_elements = find_elems('class names', driver, 'TestModeTrueFalseQuestion-prompt')
        true_false_split_text = [el.text.split('→') for el in true_false_web_elements]
        true_false_requests = [_back_words.get(lst[0].strip(), None) for lst in true_false_split_text]
        if true_false_requests[0] is None:
            true_false_requests = [_words[lst[0].strip()] for lst in true_false_split_text]
        true_false_responses = [lst[-1].strip() for lst in true_false_split_text]
        true_false_buttons_spans = find_elems('class names', driver, 'TestModeTrueFalseQuestion-choices')
        true_false_number = len(true_false_buttons_spans)
        for span in range(true_false_number):
            true_false_buttons = true_false_buttons_spans[span].find_elements_by_class_name('UIRadio')
            if true_false_requests[span] == true_false_responses[span]:
                driver.execute_script("arguments[0].click();", true_false_buttons[0])
            else:
                driver.execute_script("arguments[0].click();", true_false_buttons[1])
            time.sleep(ACTION_DELAY)
            print('[', '#' * (__i * __n // __l + 1), ' ' * (__n - __i * __n // __l - 1), ']', '\r', sep='', end='')
            __i += 1
        time.sleep(segment_delay)
        driver.execute_script("arguments[0].click();", find_elems('class names', driver, 'UIButton')[-1])
        print()
        return 0
    except (StaleElementReferenceException, NoSuchElementException, KeyboardInterrupt):
        print()
        print('\033[1;33m[ ! ] SPELL INTERRUPTED\033[0m')
        return 1


def match(driver, _words, _mod_hash):
    """
    MATCH IMPLEMENTATION
    to stop running - go to previous page or press Ctrl+C (first is preferable)
    :param driver: webdriver object
    :param _words: RU -> EN dict
    :param _mod_hash: module hash number (from URL)
    :return: (int) status code (0 if ok, else 1)
    """
    click_delay = 0.2
    prev_size = driver.get_window_size()
    driver.set_window_size(height=800, width=800)
    time.sleep(ACTION_DELAY)
    driver.get('https://quizlet.com/{}/match'.format(_mod_hash))
    find_elem('class name', driver, 'MatchModeInstructionsModal-button').click()
    timer = driver.find_element_by_class_name('MatchModeControls-currentTime')
    while int(timer.text[2]) <= 3:
        time.sleep(0.05)
    driver.execute_script('setTimeout(function(){for(var F = setTimeout(";"), i = 0; i < F; i++) clearTimeout(i)}, 100);')
    buttons = driver.find_elements_by_class_name('MatchModeQuestionGridTile-content')
    while len(buttons) > 0:
        request_button = 0
        for request_button in range(len(buttons)):
            if _words.get(buttons[request_button].text.strip()) is not None:
                break
        request = _words.get(buttons[request_button].text.strip())
        for response_button in range(len(buttons)):
            if buttons[response_button].text.strip() == request:
                buttons[request_button].click()
                time.sleep(click_delay)
                buttons[response_button].click()
                time.sleep(click_delay)
                del buttons[max(request_button, response_button)]
                del buttons[min(request_button, response_button)]
                break
    driver.set_window_size(height=prev_size['height'], width=prev_size['width'])
    return 0


def gravity(driver, _back_words, _mod_hash):
    """
    MATCH IMPLEMENTATION
    to stop running - go to previous page or press Ctrl+C (first is preferable)
    :param driver: webdriver object
    :param _back_words: EN -> RU dict
    :param _mod_hash: module hash number (from URL)
    :return: (int) status code (0 if ok, else 1)
    """
    click_delay = 0.4
    type_delay = 0.2
    miss_delay = 0.3
    driver.set_window_size(height=1000, width=1000)
    driver.get('https://quizlet.com/{}/gravity'.format(_mod_hash))
    find_elem('class name', driver, 'GravitySplashView').find_element_by_class_name('UIButton-wrapper').click()
    time.sleep(click_delay)
    find_elems('names', driver, 'difficultyLevel')[-1].click()
    time.sleep(click_delay)
    find_elem('class name', driver, 'GravityOptionsView').find_element_by_class_name('UIButton-wrapper').click()
    time.sleep(click_delay)
    find_elem('class name', driver, 'GravityDirectionsView').find_element_by_class_name('UIButton-wrapper').click()
    text_area = find_elem('class name', driver, 'js-keymaster-allow')
    while True:
        try:
            # print('.', end='')
            end_screen = driver.find_elements_by_class_name('HighscoresMessage-button')
            if len(end_screen) != 0:
                print()
                return 0
                # break
            else:
                pass
                # print('not end')
            missed_text_area = driver.find_elements_by_class_name('GravityCopyTermView-input')
            if len(missed_text_area) != 0:
                # print('-X- missed screen')
                answer = driver.find_element_by_class_name('GravityCopyTermView-definitionText').text.strip()
                missed_text_area[0].send_keys(answer)
                for i in range(15):
                    # print('waiting')
                    text_area = driver.find_elements_by_class_name('js-keymaster-allow')
                    if len(text_area) == 0:
                        time.sleep(miss_delay)
                    else:
                        text_area = text_area[0]
                        break
            asteroids = driver.find_elements_by_class_name('GravityTerm')
            for ast in asteroids:
                if ast.text == '':
                    continue
                # print('--- found ${}$, trying${}$'.format(ast.text, _back_words.get(ast.text)))
                # print('--- typing', _back_words.get(ast.text.strip(), '$$$'))
                text_area.send_keys(_back_words.get(ast.text.strip(), '') + '\n')
                # print('-V- asteroid completed')
                time.sleep(type_delay)
        except (ElementNotInteractableException, NoSuchElementException, StaleElementReferenceException):
            if not driver.current_url.endswith('gravity'):
                # print()
                print('\033[1;33m[ ! ] GRAVITY INTERRUPTED\033[0m')
                return 1
                # break
        except KeyboardInterrupt:
            # print()
            print('\033[1;33m[ ! ] GRAVITY INTERRUPTED\033[0m')
            return 1
            # break
        # try:
        #     asteroid = driver.find_element_by_class_name('GravityTerm').text.strip()
        #     if asteroid == '':
        #         print('--- asteroid not found')
        #         continue
        #     print('-V- found', asteroid)
        # except NoSuchElementException:
        #     print('--- trying missed screen found')
        #     missed_text_area = driver.find_elements_by_class_name('GravityCopyTermView-input')
        #     if len(missed_text_area) != 0:
        #         print('-X- missed screen')
        #         answer = find_elem('class name', driver, 'GravityCopyTermView-definitionText').text.strip()
        #         missed_text_area[0].send_keys(answer)
        # else:
        #     print('--- typing', _back_words.get(asteroid))
        #     text_area.send_keys(_back_words.get(asteroid) + '\n')
        #     print('-V- asteroid completed')
        #     time.sleep(type_delay)


# print(CHROME_DRIVER)
# __DRIVER__ = start_driver(CHROME_DRIVER)
__DRIVER__ = start_driver()
__DRIVER__.set_page_load_timeout(30)
print('--- driver started')
SURF_DELAY = 0.5
LOGIN_DELAY = 2
FIND_DELAY = 0.4
ACTION_DELAY = 0.2
__n = 20

print('--- trying to login to google account')
while login_google(LOGIN, PASSWORD, __DRIVER__):
    print('--- trying to login again')
print('logined in {} seconds'.format(round(time.time() - time_start, 2)))
for mod in MODULE_LINKS:
    module_time = time.time()
    mod_hash = None
    splt = mod.split('/')
    for sp in splt:
        sp = sp.strip()
        if sp.isdigit():
            mod_hash = sp
            break
    if mod_hash is None:
        print('\033[1;31m[ X ] WRONG MODULE LINK / HASH\033[0m')
        continue
    #
    print('--- trying {} module'.format(mod_hash))
    #
    print('--- getting words')
    words, back_words = get_words(__DRIVER__, mod_hash)
    print('\033[1;32m[ V ] WORDS RECEIVED, {} WORDS TO LEARN\033[0m'.format(max(len(words), len(back_words))))
    #
    # print('--- trying flashcards')
    # if not flash_cards(__DRIVER__, len(words), mod_hash):
    #     print('\033[1;32m[ V ] FLASH CARDS COMPLETED\033[0m')
    #
    print('--- trying learn')
    if not learn(__DRIVER__, words, back_words, mod_hash):
        print('\033[1;32m[ V ] LEARN COMPLETED\033[0m')
    #
    print('--- trying spell')
    if not spell(__DRIVER__, words, mod_hash):
        print('\033[1;32m[ V ] SPELL COMPLETED\033[0m')
    #
    print('--- trying test')
    if not test(__DRIVER__, words, back_words, mod_hash):
        print('\033[1;32m[ V ] TEST COMPLETED\033[0m')
    #
    print('--- trying match')
    if not match(__DRIVER__, words, mod_hash):
        print('\033[1;32m[ V ] MATCH COMPLETED\033[0m')
    #
    print('--- trying gravity')
    if not gravity(__DRIVER__, back_words, mod_hash):
        print('\033[1;32m[ V ] GRAVITY COMPLETED\033[0m')
    print('module completed in {} seconds'.format(round(time.time() - module_time, 2)))
print('program completed in {} seconds'.format(round(time.time() - time_start, 2)))
